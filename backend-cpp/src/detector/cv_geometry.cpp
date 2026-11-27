#include "cv_geometry.hpp"
#include "cv_typography.hpp"
#include "math_det_ops.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <utility>

namespace vellum::detector {

namespace {

inline uint8_t rgbToLuma(uint8_t r, uint8_t g, uint8_t b) noexcept {
    return static_cast<uint8_t>((299 * static_cast<int>(r) +
                                 587 * static_cast<int>(g) +
                                 114 * static_cast<int>(b) + 500) / 1000);
}

inline double cross2d(const Point2f& o, const Point2f& a, const Point2f& b) noexcept {
    return static_cast<double>(a.x - o.x) * static_cast<double>(b.y - o.y) -
           static_cast<double>(a.y - o.y) * static_cast<double>(b.x - o.x);
}

inline double dist2d(const Point2f& a, const Point2f& b) noexcept {
    return std::hypot(static_cast<double>(b.x - a.x), static_cast<double>(b.y - a.y));
}

inline double pointToSegDist(const Point2f& p, const Point2f& a, const Point2f& b) noexcept {
    const double vx = static_cast<double>(b.x - a.x);
    const double vy = static_cast<double>(b.y - a.y);
    const double len2 = vx * vx + vy * vy;
    if (len2 <= 1e-12) {
        return dist2d(p, a);
    }
    const double t = std::clamp(
        ((static_cast<double>(p.x - a.x) * vx) + (static_cast<double>(p.y - a.y) * vy)) / len2,
        0.0,
        1.0);
    const double proj_x = static_cast<double>(a.x) + t * vx;
    const double proj_y = static_cast<double>(a.y) + t * vy;
    return std::hypot(static_cast<double>(p.x) - proj_x, static_cast<double>(p.y) - proj_y);
}

double signedPolygonArea(const std::vector<Point2f>& poly) noexcept {
    const size_t n = poly.size();
    if (n < 3) return 0.0;
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const Point2f& a = poly[i];
        const Point2f& b = poly[(i + 1) % n];
        s += static_cast<double>(a.x) * static_cast<double>(b.y) -
             static_cast<double>(b.x) * static_cast<double>(a.y);
    }
    return 0.5 * s;
}

double polygonPerimeter(const std::vector<Point2f>& poly) noexcept {
    const size_t n = poly.size();
    if (n < 2) return 0.0;
    double perim = 0.0;
    for (size_t i = 0; i < n; ++i) {
        perim += dist2d(poly[i], poly[(i + 1) % n]);
    }
    return perim;
}

bool pointInPolygon(const Point2f& pt, const std::vector<Point2f>& poly) noexcept {
    const size_t n = poly.size();
    if (n < 3) return false;
    bool inside = false;
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        const float xi = poly[i].x, yi = poly[i].y;
        const float xj = poly[j].x, yj = poly[j].y;
        const bool intersect = ((yi > pt.y) != (yj > pt.y)) &&
                               (pt.x < (xj - xi) * (pt.y - yi) / (yj - yi + 1e-12f) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}

// Open-chain Ramer-Douglas-Peucker simplification.
void rdpOpenChain(
    const std::vector<Point2f>& pts,
    size_t i0,
    size_t i1,
    double eps,
    std::vector<uint8_t>& keep) {
    if (i1 <= i0 + 1) return;

    // Iterative stack avoids deep recursion on large contours.
    std::vector<std::pair<size_t, size_t>> stack;
    stack.reserve(64);
    stack.emplace_back(i0, i1);

    while (!stack.empty()) {
        const auto [a, b] = stack.back();
        stack.pop_back();
        if (b <= a + 1) continue;

        double max_d = -1.0;
        size_t best_i = a;
        for (size_t i = a + 1; i < b; ++i) {
            const double d = pointToSegDist(pts[i], pts[a], pts[b]);
            if (d > max_d) {
                max_d = d;
                best_i = i;
            }
        }
        if (max_d > eps && best_i > a && best_i < b) {
            keep[best_i] = 1;
            stack.emplace_back(a, best_i);
            stack.emplace_back(best_i, b);
        }
    }
}

std::vector<Point2f> simplifyClosedOnce(const std::vector<Point2f>& poly, double eps) {
    const size_t n = poly.size();
    if (n <= 3) return poly;

    // Split closed ring at vertex 0 and the vertex farthest from vertex 0.
    size_t far_idx = 1;
    double best_d = -1.0;
    for (size_t i = 1; i < n; ++i) {
        const double d = dist2d(poly[0], poly[i]);
        if (d > best_d) {
            best_d = d;
            far_idx = i;
        }
    }

    // Chain 1: 0 .. far_idx
    std::vector<Point2f> c1(poly.begin(), poly.begin() + static_cast<ptrdiff_t>(far_idx + 1));
    std::vector<uint8_t> k1(c1.size(), 0);
    k1.front() = 1;
    k1.back() = 1;
    rdpOpenChain(c1, 0, c1.size() - 1, eps, k1);

    // Chain 2: far_idx .. n-1, 0
    std::vector<Point2f> c2;
    c2.reserve(n - far_idx + 1);
    for (size_t i = far_idx; i < n; ++i) c2.push_back(poly[i]);
    c2.push_back(poly[0]);
    std::vector<uint8_t> k2(c2.size(), 0);
    k2.front() = 1;
    k2.back() = 1;
    rdpOpenChain(c2, 0, c2.size() - 1, eps, k2);

    std::vector<Point2f> out;
    out.reserve(n);
    for (size_t i = 0; i < c1.size(); ++i) {
        if (k1[i]) out.push_back(c1[i]);
    }
    for (size_t i = 1; i + 1 < c2.size(); ++i) {
        if (k2[i]) out.push_back(c2[i]);
    }
    return (out.size() >= 3) ? out : poly;
}

// Extract ordered outer boundary contour (Moore-Neighbor tracing) of a binary mask (w x h).
// Returned points are in page coordinates (offset by mask_rect.x1, mask_rect.y1).
std::vector<Point2f> extractOuterContour(
    const uint8_t* mask,
    int w,
    int h,
    int offset_x,
    int offset_y) {
    if (!mask || w <= 0 || h <= 0) return {};

    int sx = -1, sy = -1;
    for (int y = 0; y < h && sx < 0; ++y) {
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(w);
        for (int x = 0; x < w; ++x) {
            if (mask[row + static_cast<size_t>(x)] != 0) {
                sx = x;
                sy = y;
                break;
            }
        }
    }
    if (sx < 0) return {};

    auto isFg = [&](int x, int y) noexcept -> bool {
        if (x < 0 || x >= w || y < 0 || y >= h) return false;
        return mask[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)] != 0;
    };

    // Clockwise 8-neighborhood offsets starting from West (left)
    static constexpr int kDx[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
    static constexpr int kDy[8] = {0, -1, -1, -1, 0, 1, 1, 1};

    std::vector<Point2f> contour;
    contour.reserve(static_cast<size_t>(w + h) * 2ULL);

    int cx = sx;
    int cy = sy;
    int enter_dir = 0;  // Came from West (background)
    const int max_steps = std::max(64, 4 * w * h);

    for (int step = 0; step < max_steps; ++step) {
        contour.push_back(Point2f{
            static_cast<float>(offset_x + cx) + 0.5f,
            static_cast<float>(offset_y + cy) + 0.5f
        });

        int found_dir = -1;
        for (int k = 0; k < 8; ++k) {
            const int dir = (enter_dir + k) & 7;
            const int nx = cx + kDx[dir];
            const int ny = cy + kDy[dir];
            if (isFg(nx, ny)) {
                found_dir = dir;
                break;
            }
        }

        if (found_dir < 0) {
            break;  // Isolated single pixel
        }

        cx += kDx[found_dir];
        cy += kDy[found_dir];
        // Backtrack to the background pixel just prior to `found_dir`
        enter_dir = (found_dir + 5) & 7;

        if (cx == sx && cy == sy && enter_dir == 0) {
            break;
        }
        if (cx == sx && cy == sy && step > 2) {
            break;
        }
    }

    // Ensure CCW orientation for consistent hull/defect processing
    if (contour.size() >= 3 && signedPolygonArea(contour) < 0.0) {
        std::reverse(contour.begin(), contour.end());
    }
    return contour;
}

// Compute convex hull vertex indices into `poly` (ordered cyclically along `poly`).
std::vector<size_t> computeConvexHullIndices(const std::vector<Point2f>& poly) {
    const size_t n = poly.size();
    if (n < 3) return {};

    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        if (poly[a].x != poly[b].x) return poly[a].x < poly[b].x;
        return poly[a].y < poly[b].y;
    });

    std::vector<size_t> hull;
    hull.reserve(n * 2);

    // Lower hull
    for (size_t i : idx) {
        while (hull.size() >= 2 &&
               cross2d(poly[hull[hull.size() - 2]], poly[hull.back()], poly[i]) <= 0.0) {
            hull.pop_back();
        }
        hull.push_back(i);
    }

    // Upper hull
    const size_t lower_sz = hull.size();
    for (size_t k = n; k-- > 0;) {
        const size_t i = idx[k];
        while (hull.size() > lower_sz &&
               cross2d(poly[hull[hull.size() - 2]], poly[hull.back()], poly[i]) <= 0.0) {
            hull.pop_back();
        }
        hull.push_back(i);
    }

    if (!hull.empty()) hull.pop_back();  // Duplicate start vertex

    // Sort hull indices in ascending cyclic order along `poly`.
    std::sort(hull.begin(), hull.end());
    hull.erase(std::unique(hull.begin(), hull.end()), hull.end());
    return hull;
}

std::vector<std::array<int, 2>> ensureSeedPixels(const RoomSeedBlock& seed) {
    if (!seed.seed_pixels.empty()) {
        return seed.seed_pixels;
    }
    std::vector<std::array<int, 2>> pts;
    const int w = seed.bbox.width();
    const int h = seed.bbox.height();
    if (w <= 0 || h <= 0) return pts;

    const int step_x = std::max(1, w / 8);
    const int step_y = std::max(1, h / 8);
    for (int y = seed.bbox.y1 + step_y / 2; y < seed.bbox.y2; y += step_y) {
        for (int x = seed.bbox.x1 + step_x / 2; x < seed.bbox.x2; x += step_x) {
            pts.push_back({x, y});
        }
    }
    if (pts.empty()) {
        pts.push_back({(seed.bbox.x1 + seed.bbox.x2) / 2, (seed.bbox.y1 + seed.bbox.y2) / 2});
    }
    return pts;
}

// Rasterize a closed polygon into a local binary mask over `rect`.
void rasterizePolygon(
    const std::vector<Point2f>& poly,
    const RectI& rect,
    std::vector<uint8_t>& out_mask) {
    const int rw = rect.width();
    const int rh = rect.height();
    out_mask.assign(static_cast<size_t>(rw) * static_cast<size_t>(rh), 0);
    const size_t n = poly.size();
    if (n < 3 || rw <= 0 || rh <= 0) return;

    std::vector<float> x_inter;
    x_inter.reserve(32);

    for (int ly = 0; ly < rh; ++ly) {
        const float py = static_cast<float>(rect.y1 + ly) + 0.5f;
        x_inter.clear();
        for (size_t i = 0, j = n - 1; i < n; j = i++) {
            const float yi = poly[i].y;
            const float yj = poly[j].y;
            if ((yi <= py && yj > py) || (yj <= py && yi > py)) {
                const float xi = poly[i].x;
                const float xj = poly[j].x;
                x_inter.push_back(xi + (py - yi) * (xj - xi) / (yj - yi));
            }
        }
        std::sort(x_inter.begin(), x_inter.end());
        const size_t row = static_cast<size_t>(ly) * static_cast<size_t>(rw);
        for (size_t k = 0; k + 1 < x_inter.size(); k += 2) {
            const int x0 = std::clamp(static_cast<int>(std::floor(x_inter[k])) - rect.x1, 0, rw);
            const int x1 = std::clamp(static_cast<int>(std::ceil(x_inter[k + 1])) - rect.x1, 0, rw);
            for (int lx = x0; lx < x1; ++lx) {
                out_mask[row + static_cast<size_t>(lx)] = 1;
            }
        }
    }
}

// Squared-Euclidean Distance-Transform Watershed (smallest box priority) on a room polygon.
void watershedPartitionRoom(
    const std::vector<Point2f>& room_poly,
    const std::vector<size_t>& member_indices,
    const std::vector<RoomSeedBlock>& all_seeds,
    const std::vector<std::vector<std::array<int, 2>>>& all_seed_pts,
    std::vector<std::vector<Point2f>>& out_room_polys,
    double rdp_eps_ratio,
    int max_vertices) {
    if (member_indices.empty()) return;
    if (member_indices.size() == 1) {
        out_room_polys[member_indices[0]] =
            simplifyPolygonRDP(room_poly, rdp_eps_ratio, max_vertices);
        return;
    }

    // Bounding rect of room_poly
    float min_x = room_poly[0].x, max_x = room_poly[0].x;
    float min_y = room_poly[0].y, max_y = room_poly[0].y;
    for (const auto& p : room_poly) {
        if (p.x < min_x) min_x = p.x;
        if (p.x > max_x) max_x = p.x;
        if (p.y < min_y) min_y = p.y;
        if (p.y > max_y) max_y = p.y;
    }
    const RectI roi{
        static_cast<int>(std::floor(min_x)),
        static_cast<int>(std::floor(min_y)),
        static_cast<int>(std::ceil(max_x)) + 1,
        static_cast<int>(std::ceil(max_y)) + 1
    };
    const int rw = roi.width();
    const int rh = roi.height();
    if (rw <= 1 || rh <= 1) {
        for (size_t mi : member_indices) {
            out_room_polys[mi] = simplifyPolygonRDP(room_poly, rdp_eps_ratio, max_vertices);
        }
        return;
    }

    std::vector<uint8_t> room_mask;
    rasterizePolygon(room_poly, roi, room_mask);

    // Sort members smallest bbox area first (smallest-box priority).
    std::vector<size_t> ordered = member_indices;
    std::sort(ordered.begin(), ordered.end(), [&](size_t a, size_t b) {
        const int64_t aa = all_seeds[a].bbox.area();
        const int64_t ab = all_seeds[b].bbox.area();
        if (aa != ab) return aa < ab;
        return a < b;
    });

    const size_t rtotal = static_cast<size_t>(rw) * static_cast<size_t>(rh);
    std::vector<float> best_dist_sq(rtotal, std::numeric_limits<float>::infinity());
    std::vector<int> owner(rtotal, -1);

    std::vector<uint8_t> seed_fg(rtotal, 0);
    std::vector<float> dsq;

    for (size_t k = 0; k < ordered.size(); ++k) {
        const size_t mi = ordered[k];
        std::fill(seed_fg.begin(), seed_fg.end(), 0);
        int fg_count = 0;
        for (const auto& sp : all_seed_pts[mi]) {
            if (roi.contains(sp[0], sp[1])) {
                seed_fg[static_cast<size_t>(sp[1] - roi.y1) * static_cast<size_t>(rw) +
                        static_cast<size_t>(sp[0] - roi.x1)] = 1;
                ++fg_count;
            }
        }
        if (fg_count == 0) {
            const int cx = std::clamp((all_seeds[mi].bbox.x1 + all_seeds[mi].bbox.x2) / 2, roi.x1, roi.x2 - 1);
            const int cy = std::clamp((all_seeds[mi].bbox.y1 + all_seeds[mi].bbox.y2) / 2, roi.y1, roi.y2 - 1);
            seed_fg[static_cast<size_t>(cy - roi.y1) * static_cast<size_t>(rw) +
                    static_cast<size_t>(cx - roi.x1)] = 1;
        }

        if (!computeEuclideanDistanceSqFH(seed_fg.data(), rw, rh, dsq)) {
            continue;
        }

        // Strict `<` gives tie-break priority to smaller boxes processed earlier.
        for (size_t i = 0; i < rtotal; ++i) {
            if (room_mask[i] == 0) continue;
            if (dsq[i] < best_dist_sq[i]) {
                best_dist_sq[i] = dsq[i];
                owner[i] = static_cast<int>(k);
            }
        }
    }

    std::vector<uint8_t> part_mask(rtotal, 0);
    for (size_t k = 0; k < ordered.size(); ++k) {
        const size_t mi = ordered[k];
        std::fill(part_mask.begin(), part_mask.end(), 0);
        int cnt = 0;
        for (size_t i = 0; i < rtotal; ++i) {
            if (owner[i] == static_cast<int>(k)) {
                part_mask[i] = 1;
                ++cnt;
            }
        }

        std::vector<Point2f> poly;
        if (cnt >= 4) {
            poly = extractOuterContour(part_mask.data(), rw, rh, roi.x1, roi.y1);
        }
        if (poly.size() >= 3) {
            out_room_polys[mi] = simplifyPolygonRDP(poly, rdp_eps_ratio, max_vertices);
        } else {
            const auto& b = all_seeds[mi].bbox;
            out_room_polys[mi] = {
                Point2f{static_cast<float>(b.x1), static_cast<float>(b.y1)},
                Point2f{static_cast<float>(b.x2), static_cast<float>(b.y1)},
                Point2f{static_cast<float>(b.x2), static_cast<float>(b.y2)},
                Point2f{static_cast<float>(b.x1), static_cast<float>(b.y2)}
            };
        }
    }
}

// Recursive Conjoined Balloon Waist-Cutting.
void recursiveWaistCut(
    const std::vector<Point2f>& poly,
    const std::vector<size_t>& member_indices,
    const std::vector<RoomSeedBlock>& all_seeds,
    const std::vector<std::vector<std::array<int, 2>>>& all_seed_pts,
    std::vector<std::vector<Point2f>>& out_room_polys,
    double rdp_eps_ratio,
    int max_vertices,
    int depth) {
    if (member_indices.empty()) return;
    if (member_indices.size() == 1) {
        out_room_polys[member_indices[0]] =
            simplifyPolygonRDP(poly, rdp_eps_ratio, max_vertices);
        return;
    }

    const double area_total = std::abs(signedPolygonArea(poly));
    if (poly.size() < 6 || area_total < 64.0 || depth > 8) {
        watershedPartitionRoom(
            poly, member_indices, all_seeds, all_seed_pts, out_room_polys, rdp_eps_ratio, max_vertices);
        return;
    }

    // 1. Compute convex hull and convexity defects with perpendicular depth >= 0.04 * sqrt(Area).
    const std::vector<size_t> hull_idx = computeConvexHullIndices(poly);
    const double min_defect_depth = 0.04 * std::sqrt(area_total);
    std::vector<size_t> defect_vertices;

    const size_t n = poly.size();
    const size_t m = hull_idx.size();
    if (m >= 3) {
        for (size_t k = 0; k < m; ++k) {
            const size_t a = hull_idx[k];
            const size_t b = hull_idx[(k + 1) % m];
            const Point2f& pa = poly[a];
            const Point2f& pb = poly[b];
            const double edge_len = dist2d(pa, pb);
            if (edge_len <= 1e-6) continue;

            double max_d = 0.0;
            size_t best_v = a;
            for (size_t cur = (a + 1) % n; cur != b; cur = (cur + 1) % n) {
                const double d = std::abs(cross2d(pa, pb, poly[cur])) / edge_len;
                if (d > max_d) {
                    max_d = d;
                    best_v = cur;
                }
            }
            if (max_d >= min_defect_depth && best_v != a && best_v != b) {
                defect_vertices.push_back(best_v);
            }
        }
    }

    // 2. Build candidate chords sorted by `chord_length / sqrt(min(A1, A_total - A1))`.
    struct CandidateChord {
        size_t i1 = 0;
        size_t i2 = 0;
        double score = 0.0;
    };
    std::vector<CandidateChord> candidates;

    auto addCandidate = [&](size_t u, size_t v) {
        if (u == v) return;
        size_t i1 = std::min(u, v);
        size_t i2 = std::max(u, v);
        if (i2 - i1 < 2 || (n - (i2 - i1)) < 2) return;

        std::vector<Point2f> sub1;
        sub1.reserve(i2 - i1 + 1);
        for (size_t i = i1; i <= i2; ++i) sub1.push_back(poly[i]);
        const double a1 = std::abs(signedPolygonArea(sub1));
        const double a2 = std::max(0.0, area_total - a1);
        const double min_a = std::min(a1, a2);
        if (min_a < 0.05 * area_total) return;

        const double chord_len = dist2d(poly[i1], poly[i2]);
        if (chord_len <= 1.0) return;
        const double score = chord_len / std::sqrt(min_a);
        candidates.push_back(CandidateChord{i1, i2, score});
    };

    for (size_t a = 0; a < defect_vertices.size(); ++a) {
        for (size_t b = a + 1; b < defect_vertices.size(); ++b) {
            addCandidate(defect_vertices[a], defect_vertices[b]);
        }
    }

    // If only 1 deep defect exists (single-sided pinch), pair it with nearest opposite contour vertices.
    if (defect_vertices.size() == 1) {
        const size_t dv = defect_vertices[0];
        const size_t stride = std::max<size_t>(1, n / 32);
        for (size_t v = 0; v < n; v += stride) {
            const size_t cyc_dist = std::min((v + n - dv) % n, (dv + n - v) % n);
            if (cyc_dist >= n / 6) {
                addCandidate(dv, v);
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const CandidateChord& a, const CandidateChord& b) {
        return a.score < b.score;
    });

    // 3. Test candidate chords: inside bubble AND separating >= 90% (SIDE = 0.90) of each block's seeds.
    constexpr double kSideThreshold = 0.90;

    for (const auto& cand : candidates) {
        const Point2f& p1 = poly[cand.i1];
        const Point2f& p2 = poly[cand.i2];

        // Verify chord interior lies inside `poly`.
        bool chord_inside = true;
        for (int s = 1; s <= 7; ++s) {
            const float t = static_cast<float>(s) / 8.0f;
            const Point2f mid{
                p1.x + t * (p2.x - p1.x),
                p1.y + t * (p2.y - p1.y)
            };
            if (!pointInPolygon(mid, poly)) {
                chord_inside = false;
                break;
            }
        }
        if (!chord_inside) continue;

        std::vector<size_t> side_pos;
        std::vector<size_t> side_neg;
        bool clean_split = true;

        for (size_t mi : member_indices) {
            const auto& pts = all_seed_pts[mi];
            if (pts.empty()) {
                clean_split = false;
                break;
            }
            int pos_cnt = 0;
            int neg_cnt = 0;
            for (const auto& sp : pts) {
                const Point2f q{static_cast<float>(sp[0]) + 0.5f, static_cast<float>(sp[1]) + 0.5f};
                const double c = cross2d(p1, p2, q);
                if (c >= 0.0) ++pos_cnt;
                if (c <= 0.0) ++neg_cnt;
            }
            const double f_pos = static_cast<double>(pos_cnt) / static_cast<double>(pts.size());
            const double f_neg = static_cast<double>(neg_cnt) / static_cast<double>(pts.size());
            if (f_pos >= kSideThreshold && f_pos >= f_neg) {
                side_pos.push_back(mi);
            } else if (f_neg >= kSideThreshold) {
                side_neg.push_back(mi);
            } else {
                clean_split = false;
                break;
            }
        }

        if (!clean_split || side_pos.empty() || side_neg.empty()) {
            continue;
        }

        // Split `poly` into two sub-polygons along (cand.i1, cand.i2).
        std::vector<Point2f> poly_a;
        poly_a.reserve(cand.i2 - cand.i1 + 1);
        for (size_t i = cand.i1; i <= cand.i2; ++i) poly_a.push_back(poly[i]);

        std::vector<Point2f> poly_b;
        poly_b.reserve(n - (cand.i2 - cand.i1) + 1);
        for (size_t i = cand.i2; i < n; ++i) poly_b.push_back(poly[i]);
        for (size_t i = 0; i <= cand.i1; ++i) poly_b.push_back(poly[i]);

        // Match side_pos / side_neg to poly_a / poly_b using centroid containment.
        auto countInPoly = [&](const std::vector<size_t>& group, const std::vector<Point2f>& sub) {
            int hits = 0;
            for (size_t mi : group) {
                const auto& b = all_seeds[mi].bbox;
                const Point2f ctr{0.5f * static_cast<float>(b.x1 + b.x2),
                                  0.5f * static_cast<float>(b.y1 + b.y2)};
                if (pointInPolygon(ctr, sub)) ++hits;
            }
            return hits;
        };

        const int match_direct = countInPoly(side_pos, poly_a) + countInPoly(side_neg, poly_b);
        const int match_swapped = countInPoly(side_pos, poly_b) + countInPoly(side_neg, poly_a);

        const auto& grp_a = (match_direct >= match_swapped) ? side_pos : side_neg;
        const auto& grp_b = (match_direct >= match_swapped) ? side_neg : side_pos;

        recursiveWaistCut(poly_a, grp_a, all_seeds, all_seed_pts, out_room_polys, rdp_eps_ratio, max_vertices, depth + 1);
        recursiveWaistCut(poly_b, grp_b, all_seeds, all_seed_pts, out_room_polys, rdp_eps_ratio, max_vertices, depth + 1);
        return;
    }

    // No valid waist chord found -> fall back to squared-Euclidean EDT watershed.
    watershedPartitionRoom(
        poly, member_indices, all_seeds, all_seed_pts, out_room_polys, rdp_eps_ratio, max_vertices);
}

// -----------------------------------------------------------------------------
// Panel-First Reading Order Helpers
// -----------------------------------------------------------------------------

struct BoxItem {
    size_t orig_idx = 0;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;

    float cx() const noexcept { return 0.5f * (x1 + x2); }
    float cy() const noexcept { return 0.5f * (y1 + y2); }
};

BoxItem erodeBox(const BoxItem& b, float frac) noexcept {
    const float dx = 0.5f * frac * std::max(0.0f, b.x2 - b.x1);
    const float dy = 0.5f * frac * std::max(0.0f, b.y2 - b.y1);
    BoxItem out = b;
    out.x1 += dx;
    out.x2 -= dx;
    out.y1 += dy;
    out.y2 -= dy;
    if (out.x2 < out.x1) {
        const float mx = b.cx();
        out.x1 = mx;
        out.x2 = mx;
    }
    if (out.y2 < out.y1) {
        const float my = b.cy();
        out.y1 = my;
        out.y2 = my;
    }
    return out;
}

// Try to find a clean horizontal (y) or vertical (x) guillotine cut on `boxes`.
// Returns true if split into two non-empty subsets `first_half` and `second_half`
// already ordered according to top-to-bottom / RTL / LTR.
bool tryGuillotineSplit(
    const std::vector<BoxItem>& boxes,
    bool rtl,
    std::vector<size_t>& first_half,
    std::vector<size_t>& second_half) {
    const size_t n = boxes.size();
    if (n <= 1) return false;

    // 1. Horizontal guillotine cut (top-to-bottom)
    {
        std::vector<size_t> by_y(n);
        std::iota(by_y.begin(), by_y.end(), 0);
        std::sort(by_y.begin(), by_y.end(), [&](size_t a, size_t b) {
            if (boxes[a].y1 != boxes[b].y1) return boxes[a].y1 < boxes[b].y1;
            return boxes[a].y2 < boxes[b].y2;
        });

        float max_y2 = boxes[by_y[0]].y2;
        for (size_t i = 0; i + 1 < n; ++i) {
            max_y2 = std::max(max_y2, boxes[by_y[i]].y2);
            if (max_y2 <= boxes[by_y[i + 1]].y1) {
                first_half.assign(by_y.begin(), by_y.begin() + static_cast<ptrdiff_t>(i + 1));
                second_half.assign(by_y.begin() + static_cast<ptrdiff_t>(i + 1), by_y.end());
                return true;
            }
        }
    }

    // 2. Vertical guillotine cut (RTL or LTR)
    {
        std::vector<size_t> by_x(n);
        std::iota(by_x.begin(), by_x.end(), 0);
        std::sort(by_x.begin(), by_x.end(), [&](size_t a, size_t b) {
            if (boxes[a].x1 != boxes[b].x1) return boxes[a].x1 < boxes[b].x1;
            return boxes[a].x2 < boxes[b].x2;
        });

        float max_x2 = boxes[by_x[0]].x2;
        for (size_t i = 0; i + 1 < n; ++i) {
            max_x2 = std::max(max_x2, boxes[by_x[i]].x2);
            if (max_x2 <= boxes[by_x[i + 1]].x1) {
                if (rtl) {
                    first_half.assign(by_x.begin() + static_cast<ptrdiff_t>(i + 1), by_x.end());
                    second_half.assign(by_x.begin(), by_x.begin() + static_cast<ptrdiff_t>(i + 1));
                } else {
                    first_half.assign(by_x.begin(), by_x.begin() + static_cast<ptrdiff_t>(i + 1));
                    second_half.assign(by_x.begin() + static_cast<ptrdiff_t>(i + 1), by_x.end());
                }
                return true;
            }
        }
    }

    return false;
}

// Pairwise tournament with progressive box erosion (0%, 10%, 20%, 30%, 40%, 50%).
std::vector<size_t> sortBoxesGuillotineOrTournament(
    const std::vector<BoxItem>& items,
    bool rtl) {
    const size_t n = items.size();
    if (n == 0) return {};
    if (n == 1) return {items[0].orig_idx};

    // Progressive erosion fractions: 0%, 10%, 20%, 30%, 40%, 50%
    static constexpr std::array<float, 6> kErosions = {0.0f, 0.10f, 0.20f, 0.30f, 0.40f, 0.50f};

    for (float ef : kErosions) {
        std::vector<BoxItem> eroded(n);
        for (size_t i = 0; i < n; ++i) {
            eroded[i] = (ef > 0.0f) ? erodeBox(items[i], ef) : items[i];
        }

        std::vector<size_t> first_idx, second_idx;
        if (tryGuillotineSplit(eroded, rtl, first_idx, second_idx)) {
            std::vector<BoxItem> sub1, sub2;
            sub1.reserve(first_idx.size());
            sub2.reserve(second_idx.size());
            for (size_t i : first_idx) sub1.push_back(items[i]);
            for (size_t i : second_idx) sub2.push_back(items[i]);

            auto out1 = sortBoxesGuillotineOrTournament(sub1, rtl);
            auto out2 = sortBoxesGuillotineOrTournament(sub2, rtl);
            out1.insert(out1.end(), out2.begin(), out2.end());
            return out1;
        }
    }

    // Fallback: pairwise tournament accumulated across progressive erosions (0%..50%).
    std::vector<std::vector<int>> adj(n);
    std::vector<int> in_degree(n, 0);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            int winner = 0;  // +1 => i before j, -1 => j before i
            for (float ef : kErosions) {
                const BoxItem a = (ef > 0.0f) ? erodeBox(items[i], ef) : items[i];
                const BoxItem b = (ef > 0.0f) ? erodeBox(items[j], ef) : items[j];
                if (a.y2 <= b.y1) { winner = +1; break; }
                if (b.y2 <= a.y1) { winner = -1; break; }
                if (rtl) {
                    if (a.x1 >= b.x2) { winner = +1; break; }
                    if (b.x1 >= a.x2) { winner = -1; break; }
                } else {
                    if (a.x2 <= b.x1) { winner = +1; break; }
                    if (b.x2 <= a.x1) { winner = -1; break; }
                }
            }
            if (winner == 0) {
                // Centroid fallback
                const float dy = items[i].cy() - items[j].cy();
                const float dx = items[i].cx() - items[j].cx();
                if (std::abs(dy) >= std::abs(dx)) {
                    winner = (dy < 0.0f) ? +1 : -1;
                } else {
                    winner = rtl ? ((dx > 0.0f) ? +1 : -1) : ((dx < 0.0f) ? +1 : -1);
                }
            }
            if (winner > 0) {
                adj[i].push_back(static_cast<int>(j));
                ++in_degree[j];
            } else {
                adj[j].push_back(static_cast<int>(i));
                ++in_degree[i];
            }
        }
    }

    auto tieBreak = [&](size_t a, size_t b) noexcept {
        if (std::abs(items[a].cy() - items[b].cy()) > 1e-3f) {
            return items[a].cy() < items[b].cy();
        }
        if (std::abs(items[a].cx() - items[b].cx()) > 1e-3f) {
            return rtl ? (items[a].cx() > items[b].cx()) : (items[a].cx() < items[b].cx());
        }
        return items[a].orig_idx < items[b].orig_idx;
    };

    std::vector<uint8_t> visited(n, 0);
    std::vector<size_t> ordered;
    ordered.reserve(n);

    while (ordered.size() < n) {
        size_t best = n;
        int best_deg = 0;
        for (size_t i = 0; i < n; ++i) {
            if (visited[i]) continue;
            const int deg = std::max(0, in_degree[i]);
            if (best == n || deg < best_deg || (deg == best_deg && tieBreak(i, best))) {
                best = i;
                best_deg = deg;
            }
        }
        if (best == n) break;
        visited[best] = 1;
        ordered.push_back(items[best].orig_idx);
        for (int nxt : adj[best]) {
            if (!visited[static_cast<size_t>(nxt)] && in_degree[static_cast<size_t>(nxt)] > 0) {
                --in_degree[static_cast<size_t>(nxt)];
            }
        }
    }

    return ordered;
}

double rectIoA(const RectI& inner, const RectI& outer) noexcept {
    const int64_t ia = inner.area();
    if (ia <= 0) return 0.0;
    const RectI inter = inner.intersected(outer);
    return static_cast<double>(inter.area()) / static_cast<double>(ia);
}

}  // namespace

// -----------------------------------------------------------------------------
// Public API Implementations
// -----------------------------------------------------------------------------

std::vector<Point2f> simplifyPolygonRDP(
    const std::vector<Point2f>& closed_poly,
    double eps_ratio,
    int max_vertices) {
    if (closed_poly.size() <= 3) return closed_poly;
    const int cap = std::max(3, max_vertices);

    const double perim = polygonPerimeter(closed_poly);
    double eps = std::max(0.25, eps_ratio * perim);
    std::vector<Point2f> simp = simplifyClosedOnce(closed_poly, eps);

    // Increase epsilon progressively if vertex count still exceeds `max_vertices` (32).
    int guard = 0;
    while (static_cast<int>(simp.size()) > cap && guard < 24) {
        eps *= 1.5;
        simp = simplifyClosedOnce(closed_poly, eps);
        ++guard;
    }

    // Hard safety cap if degenerate collinear loop resisted epsilon growth
    if (static_cast<int>(simp.size()) > cap) {
        std::vector<Point2f> decimated;
        decimated.reserve(static_cast<size_t>(cap));
        const double step = static_cast<double>(simp.size()) / static_cast<double>(cap);
        for (int i = 0; i < cap; ++i) {
            const size_t idx = std::min(simp.size() - 1, static_cast<size_t>(std::floor(i * step)));
            decimated.push_back(simp[idx]);
        }
        return decimated;
    }

    return simp;
}

std::vector<std::vector<Point2f>> partitionBubbleRooms(
    const uint8_t* bubble_mask,
    const RectI& mask_rect,
    const std::vector<RoomSeedBlock>& seeds,
    bool enable_waist_cut,
    double rdp_eps_ratio,
    int max_vertices) {
    std::vector<std::vector<Point2f>> result(seeds.size());
    if (seeds.empty()) return result;

    std::vector<Point2f> contour;
    if (bubble_mask && !mask_rect.empty()) {
        contour = extractOuterContour(
            bubble_mask, mask_rect.width(), mask_rect.height(), mask_rect.x1, mask_rect.y1);
    }

    if (contour.size() < 3) {
        for (size_t i = 0; i < seeds.size(); ++i) {
            const auto& b = seeds[i].bbox;
            result[i] = {
                Point2f{static_cast<float>(b.x1), static_cast<float>(b.y1)},
                Point2f{static_cast<float>(b.x2), static_cast<float>(b.y1)},
                Point2f{static_cast<float>(b.x2), static_cast<float>(b.y2)},
                Point2f{static_cast<float>(b.x1), static_cast<float>(b.y2)}
            };
        }
        return result;
    }

    if (seeds.size() == 1) {
        result[0] = simplifyPolygonRDP(contour, rdp_eps_ratio, max_vertices);
        return result;
    }

    // Pre-simplify dense pixel contour slightly (0.5px) for fast hull/defect/chord checks.
    std::vector<Point2f> work_poly =
        (contour.size() > 256) ? simplifyClosedOnce(contour, 0.75) : contour;

    std::vector<std::vector<std::array<int, 2>>> seed_pts(seeds.size());
    for (size_t i = 0; i < seeds.size(); ++i) {
        seed_pts[i] = ensureSeedPixels(seeds[i]);
    }

    std::vector<size_t> all_idx(seeds.size());
    std::iota(all_idx.begin(), all_idx.end(), 0);

    if (enable_waist_cut) {
        recursiveWaistCut(
            work_poly, all_idx, seeds, seed_pts, result, rdp_eps_ratio, max_vertices, 0);
    } else {
        watershedPartitionRoom(
            work_poly, all_idx, seeds, seed_pts, result, rdp_eps_ratio, max_vertices);
    }

    return result;
}

std::vector<std::vector<Point2f>> computePageRoomPolygons(
    const uint8_t* rgb_data,
    int page_w,
    int page_h,
    const std::vector<RoomSeedBlock>& text_blocks,
    const std::vector<RectI>& bubble_regions,
    const std::vector<uint8_t>* text_mask,
    bool enable_waist_cut) {
    const size_t n = text_blocks.size();
    std::vector<std::vector<Point2f>> out(n);
    if (n == 0) return out;

    auto makeBoxPoly = [](const RectI& b) -> std::vector<Point2f> {
        return {
            Point2f{static_cast<float>(b.x1), static_cast<float>(b.y1)},
            Point2f{static_cast<float>(b.x2), static_cast<float>(b.y1)},
            Point2f{static_cast<float>(b.x2), static_cast<float>(b.y2)},
            Point2f{static_cast<float>(b.x1), static_cast<float>(b.y2)}
        };
    };

    if (!rgb_data || page_w <= 0 || page_h <= 0) {
        for (size_t i = 0; i < n; ++i) out[i] = makeBoxPoly(text_blocks[i].bbox);
        return out;
    }

    // Group text blocks by their best enclosing RF-DETR speech bubble region (IoA > 0.50).
    std::unordered_map<int, std::vector<size_t>> bubble_groups;
    std::vector<size_t> ungrouped;

    for (size_t i = 0; i < n; ++i) {
        int best_b = -1;
        double best_ioa = 0.50;
        int64_t best_area = std::numeric_limits<int64_t>::max();
        for (size_t b = 0; b < bubble_regions.size(); ++b) {
            const double ioa = rectIoA(text_blocks[i].bbox, bubble_regions[b]);
            if (ioa <= 0.50) continue;
            const int64_t ba = bubble_regions[b].area();
            if (ioa > best_ioa + 0.05 || (std::abs(ioa - best_ioa) <= 0.05 && ba < best_area)) {
                best_ioa = ioa;
                best_area = ba;
                best_b = static_cast<int>(b);
            }
        }
        if (best_b >= 0) {
            bubble_groups[best_b].push_back(i);
        } else {
            ungrouped.push_back(i);
        }
    }

    const bool has_tmask =
        (text_mask != nullptr &&
         text_mask->size() >= static_cast<size_t>(page_w) * static_cast<size_t>(page_h));

    // Helper: flood-fill bubble interior inside `roi` starting from seed boxes' bright/paper pixels
    // (treating `text_mask` pixels as passable so text strokes never block interior flood-fill).
    auto extractBubbleInteriorMask = [&](const RectI& roi,
                                         const std::vector<size_t>& members,
                                         std::vector<uint8_t>& mask_out) -> bool {
        const int rw = roi.width();
        const int rh = roi.height();
        if (rw < 8 || rh < 8) return false;
        const size_t rtotal = static_cast<size_t>(rw) * static_cast<size_t>(rh);

        // Determine interior paper luma from non-text pixels inside member bboxes.
        std::array<uint32_t, 256> hist{};
        uint32_t cnt = 0;
        for (size_t mi : members) {
            const RectI tb = text_blocks[mi].bbox.intersected(roi);
            for (int gy = tb.y1; gy < tb.y2; ++gy) {
                const size_t prow = static_cast<size_t>(gy) * static_cast<size_t>(page_w);
                for (int gx = tb.x1; gx < tb.x2; ++gx) {
                    if (has_tmask && (*text_mask)[prow + static_cast<size_t>(gx)] != 0) continue;
                    const uint8_t* px = rgb_data + (prow + static_cast<size_t>(gx)) * 3ULL;
                    ++hist[rgbToLuma(px[0], px[1], px[2])];
                    ++cnt;
                }
            }
        }
        if (cnt == 0) return false;

        uint32_t acc = 0;
        int med_luma = 255;
        for (int i = 0; i < 256; ++i) {
            acc += hist[static_cast<size_t>(i)];
            if (acc > cnt / 2) { med_luma = i; break; }
        }
        const bool dark_bubble = (med_luma < 90);

        auto isPassable = [&](int gx, int gy) noexcept -> bool {
            const size_t pidx = static_cast<size_t>(gy) * static_cast<size_t>(page_w) + static_cast<size_t>(gx);
            if (has_tmask && (*text_mask)[pidx] != 0) return true;
            const uint8_t* px = rgb_data + pidx * 3ULL;
            const int luma = static_cast<int>(rgbToLuma(px[0], px[1], px[2]));
            return dark_bubble ? (luma < 140) : (luma >= 115);
        };

        mask_out.assign(rtotal, 0);
        std::vector<int> queue;
        queue.reserve(rtotal / 2);

        for (size_t mi : members) {
            const RectI tb = text_blocks[mi].bbox.intersected(roi);
            for (int gy = tb.y1; gy < tb.y2; ++gy) {
                for (int gx = tb.x1; gx < tb.x2; ++gx) {
                    const int lx = gx - roi.x1;
                    const int ly = gy - roi.y1;
                    const int idx = ly * rw + lx;
                    if (mask_out[static_cast<size_t>(idx)] == 0 && isPassable(gx, gy)) {
                        mask_out[static_cast<size_t>(idx)] = 1;
                        queue.push_back(idx);
                    }
                }
            }
        }

        if (queue.empty()) return false;

        static constexpr int kDx4[4] = {-1, 1, 0, 0};
        static constexpr int kDy4[4] = {0, 0, -1, 1};
        size_t head = 0;
        while (head < queue.size()) {
            const int cur = queue[head++];
            const int cx = cur % rw;
            const int cy = cur / rw;
            for (int k = 0; k < 4; ++k) {
                const int nx = cx + kDx4[k];
                const int ny = cy + kDy4[k];
                if (nx < 0 || nx >= rw || ny < 0 || ny >= rh) continue;
                const int nidx = ny * rw + nx;
                if (mask_out[static_cast<size_t>(nidx)] == 0 && isPassable(roi.x1 + nx, roi.y1 + ny)) {
                    mask_out[static_cast<size_t>(nidx)] = 1;
                    queue.push_back(nidx);
                }
            }
        }

        return queue.size() >= 16;
    };

    // Process each grouped bubble region
    for (const auto& [b_idx, members] : bubble_groups) {
        const RectI roi = bubble_regions[static_cast<size_t>(b_idx)].clamped(page_w, page_h);
        std::vector<uint8_t> bmask;
        const bool ok = extractBubbleInteriorMask(roi, members, bmask);

        std::vector<RoomSeedBlock> local_seeds;
        local_seeds.reserve(members.size());
        for (size_t mi : members) local_seeds.push_back(text_blocks[mi]);

        if (ok) {
            auto polys = partitionBubbleRooms(
                bmask.data(), roi, local_seeds, enable_waist_cut, 0.001, 32);
            for (size_t k = 0; k < members.size(); ++k) {
                out[members[k]] = std::move(polys[k]);
            }
        } else {
            for (size_t mi : members) {
                out[mi] = makeBoxPoly(text_blocks[mi].bbox);
            }
        }
    }

    // Ungrouped (borderless / CTD) blocks get their tight bbox polygon
    for (size_t mi : ungrouped) {
        out[mi] = makeBoxPoly(text_blocks[mi].bbox);
    }

    return out;
}

std::vector<RectI> detectPanelsClassicCV(
    const uint8_t* rgb_data,
    int page_w,
    int page_h) {
    if (!rgb_data || page_w < 32 || page_h < 32) return {};

    // Downsample so long side <= 512 px -> runs in ~1-2 ms!
    const int long_side = std::max(page_w, page_h);
    const int ds = std::max(1, (long_side + 511) / 512);
    const int sw = std::max(1, page_w / ds);
    const int sh = std::max(1, page_h / ds);
    const size_t stotal = static_cast<size_t>(sw) * static_cast<size_t>(sh);

    std::vector<uint8_t> gray(stotal, 255);
    for (int sy = 0; sy < sh; ++sy) {
        const int gy = std::min(page_h - 1, sy * ds + ds / 2);
        const size_t rgb_row = static_cast<size_t>(gy) * static_cast<size_t>(page_w) * 3ULL;
        const size_t srow = static_cast<size_t>(sy) * static_cast<size_t>(sw);
        for (int sx = 0; sx < sw; ++sx) {
            const int gx = std::min(page_w - 1, sx * ds + ds / 2);
            const uint8_t* px = rgb_data + rgb_row + static_cast<size_t>(gx) * 3ULL;
            gray[srow + static_cast<size_t>(sx)] = rgbToLuma(px[0], px[1], px[2]);
        }
    }

    // Determine gutter polarity from 2% page border strip
    std::array<uint32_t, 256> border_hist{};
    uint32_t border_cnt = 0;
    const int bw_strip = std::max(2, sw / 50);
    const int bh_strip = std::max(2, sh / 50);
    for (int y = 0; y < sh; ++y) {
        const bool y_edge = (y < bh_strip || y >= sh - bh_strip);
        const size_t srow = static_cast<size_t>(y) * static_cast<size_t>(sw);
        for (int x = 0; x < sw; ++x) {
            if (y_edge || x < bw_strip || x >= sw - bw_strip) {
                ++border_hist[gray[srow + static_cast<size_t>(x)]];
                ++border_cnt;
            }
        }
    }
    int med_border = 255;
    uint32_t acc = 0;
    for (int i = 0; i < 256; ++i) {
        acc += border_hist[static_cast<size_t>(i)];
        if (acc > border_cnt / 2) { med_border = i; break; }
    }
    const bool black_gutter = (med_border <= 75);

    std::vector<uint8_t> content(stotal, 0);
    for (size_t i = 0; i < stotal; ++i) {
        const bool is_gutter = black_gutter ? (gray[i] <= 40) : (gray[i] >= 225);
        content[i] = is_gutter ? 0 : 1;
    }

    // Flood-fill exterior gutter so panel interiors (including white speech bubbles inside black panel borders)
    // become solid panel regions!
    std::vector<uint8_t> exterior_gutter(stotal, 0);
    std::vector<int> queue;
    queue.reserve(stotal / 4);
    auto pushExt = [&](int x, int y) {
        const int idx = y * sw + x;
        if (content[static_cast<size_t>(idx)] == 0 && exterior_gutter[static_cast<size_t>(idx)] == 0) {
            exterior_gutter[static_cast<size_t>(idx)] = 1;
            queue.push_back(idx);
        }
    };
    for (int x = 0; x < sw; ++x) { pushExt(x, 0); pushExt(x, sh - 1); }
    for (int y = 1; y < sh - 1; ++y) { pushExt(0, y); pushExt(sw - 1, y); }

    static constexpr int kDx4[4] = {-1, 1, 0, 0};
    static constexpr int kDy4[4] = {0, 0, -1, 1};
    size_t head = 0;
    while (head < queue.size()) {
        const int cur = queue[head++];
        const int cx = cur % sw;
        const int cy = cur / sw;
        for (int k = 0; k < 4; ++k) {
            const int nx = cx + kDx4[k];
            const int ny = cy + kDy4[k];
            if (nx >= 0 && nx < sw && ny >= 0 && ny < sh) pushExt(nx, ny);
        }
    }

    // Solid panels = everything not reached by exterior gutter flood-fill
    std::vector<uint8_t> solid(stotal, 0);
    for (size_t i = 0; i < stotal; ++i) {
        solid[i] = (exterior_gutter[i] == 0) ? 1 : 0;
    }

    // Extract 4-connected components of `solid` as candidate panel rectangles.
    const int min_pw = std::max(8, sw / 10);
    const int min_ph = std::max(8, sh / 12);
    const int64_t min_parea = static_cast<int64_t>(stotal) / 50;  // >= 2% of page

    std::vector<uint8_t> visited(stotal, 0);
    std::vector<RectI> small_panels;

    for (int y = 0; y < sh; ++y) {
        for (int x = 0; x < sw; ++x) {
            const int start = y * sw + x;
            if (solid[static_cast<size_t>(start)] == 0 || visited[static_cast<size_t>(start)] != 0) {
                continue;
            }
            queue.clear();
            queue.push_back(start);
            visited[static_cast<size_t>(start)] = 1;
            int x1 = x, x2 = x, y1 = y, y2 = y;

            head = 0;
            while (head < queue.size()) {
                const int cur = queue[head++];
                const int cx = cur % sw;
                const int cy = cur / sw;
                if (cx < x1) x1 = cx;
                if (cx > x2) x2 = cx;
                if (cy < y1) y1 = cy;
                if (cy > y2) y2 = cy;
                for (int k = 0; k < 4; ++k) {
                    const int nx = cx + kDx4[k];
                    const int ny = cy + kDy4[k];
                    if (nx < 0 || nx >= sw || ny < 0 || ny >= sh) continue;
                    const int nidx = ny * sw + nx;
                    if (solid[static_cast<size_t>(nidx)] != 0 && visited[static_cast<size_t>(nidx)] == 0) {
                        visited[static_cast<size_t>(nidx)] = 1;
                        queue.push_back(nidx);
                    }
                }
            }

            const int pw = x2 - x1 + 1;
            const int ph = y2 - y1 + 1;
            if (pw >= min_pw && ph >= min_ph && static_cast<int64_t>(pw) * static_cast<int64_t>(ph) >= min_parea) {
                small_panels.push_back(RectI{x1, y1, x2 + 1, y2 + 1});
            }
        }
    }

    // Gutter-projection recursive split on any oversized panel spanning > 55% of page.
    std::vector<RectI> split_panels;
    std::vector<RectI> work = std::move(small_panels);
    while (!work.empty()) {
        const RectI r = work.back();
        work.pop_back();
        const int rw = r.width();
        const int rh = r.height();
        bool did_split = false;

        // Try horizontal gutter projection split inside `r`
        if (rh > sh / 3) {
            int run_start = -1;
            int best_y = -1;
            int best_run = 0;
            for (int y = r.y1 + min_ph; y < r.y2 - min_ph; ++y) {
                const size_t srow = static_cast<size_t>(y) * static_cast<size_t>(sw);
                int gutter_px = 0;
                for (int x = r.x1; x < r.x2; ++x) {
                    if (content[srow + static_cast<size_t>(x)] == 0) ++gutter_px;
                }
                if (gutter_px >= static_cast<int>(0.94 * static_cast<double>(rw))) {
                    if (run_start < 0) run_start = y;
                } else if (run_start >= 0) {
                    const int run = y - run_start;
                    if (run >= 2 && run > best_run) {
                        best_run = run;
                        best_y = run_start + run / 2;
                    }
                    run_start = -1;
                }
            }
            if (best_y > r.y1 + min_ph && best_y < r.y2 - min_ph) {
                work.push_back(RectI{r.x1, r.y1, r.x2, best_y});
                work.push_back(RectI{r.x1, best_y, r.x2, r.y2});
                did_split = true;
            }
        }

        // Try vertical gutter projection split inside `r`
        if (!did_split && rw > sw / 3) {
            int run_start = -1;
            int best_x = -1;
            int best_run = 0;
            for (int x = r.x1 + min_pw; x < r.x2 - min_pw; ++x) {
                int gutter_px = 0;
                for (int y = r.y1; y < r.y2; ++y) {
                    if (content[static_cast<size_t>(y) * static_cast<size_t>(sw) + static_cast<size_t>(x)] == 0) {
                        ++gutter_px;
                    }
                }
                if (gutter_px >= static_cast<int>(0.94 * static_cast<double>(rh))) {
                    if (run_start < 0) run_start = x;
                } else if (run_start >= 0) {
                    const int run = x - run_start;
                    if (run >= 2 && run > best_run) {
                        best_run = run;
                        best_x = run_start + run / 2;
                    }
                    run_start = -1;
                }
            }
            if (best_x > r.x1 + min_pw && best_x < r.x2 - min_pw) {
                work.push_back(RectI{r.x1, r.y1, best_x, r.y2});
                work.push_back(RectI{best_x, r.y1, r.x2, r.y2});
                did_split = true;
            }
        }

        if (!did_split) {
            split_panels.push_back(r);
        }
    }

    // Scale back to native page coordinates
    std::vector<RectI> out;
    out.reserve(split_panels.size());
    for (const auto& r : split_panels) {
        const RectI native = RectI{r.x1 * ds, r.y1 * ds, r.x2 * ds, r.y2 * ds}.clamped(page_w, page_h);
        if (!native.empty()) {
            out.push_back(native);
        }
    }
    return out;
}

std::vector<size_t> sortReadingOrderPanelFirst(
    const std::vector<RectI>& text_boxes,
    const std::vector<RectI>& panel_boxes,
    bool rtl) {
    const size_t nt = text_boxes.size();
    if (nt <= 1) {
        std::vector<size_t> trivial(nt);
        std::iota(trivial.begin(), trivial.end(), 0);
        return trivial;
    }

    if (panel_boxes.empty()) {
        std::vector<BoxItem> items(nt);
        for (size_t i = 0; i < nt; ++i) {
            items[i] = BoxItem{
                i,
                static_cast<float>(text_boxes[i].x1),
                static_cast<float>(text_boxes[i].y1),
                static_cast<float>(text_boxes[i].x2),
                static_cast<float>(text_boxes[i].y2)
            };
        }
        return sortBoxesGuillotineOrTournament(items, rtl);
    }

    // 1. Nest panels with >= 90% containment.
    const size_t np = panel_boxes.size();
    std::vector<int> panel_parent(np, -1);
    std::vector<std::vector<size_t>> panel_children(np);

    for (size_t i = 0; i < np; ++i) {
        const int64_t ai = panel_boxes[i].area();
        if (ai <= 0) continue;
        int best_parent = -1;
        int64_t best_parent_area = std::numeric_limits<int64_t>::max();
        for (size_t j = 0; j < np; ++j) {
            if (i == j) continue;
            const int64_t aj = panel_boxes[j].area();
            if (aj <= ai) continue;
            if (rectIoA(panel_boxes[i], panel_boxes[j]) >= 0.90) {
                if (aj < best_parent_area) {
                    best_parent_area = aj;
                    best_parent = static_cast<int>(j);
                }
            }
        }
        panel_parent[i] = best_parent;
        if (best_parent >= 0) {
            panel_children[static_cast<size_t>(best_parent)].push_back(i);
        }
    }

    // 2. Assign text blocks to panels:
    //    - >= 25% overlap (preferring smallest enclosing/nested panel), OR
    //    - >= 50% vertical alignment beside nearest panel.
    std::vector<std::vector<size_t>> panel_texts(np);
    std::vector<size_t> unassigned_texts;

    for (size_t ti = 0; ti < nt; ++ti) {
        const RectI& tb = text_boxes[ti];
        int best_p = -1;
        double best_ioa = 0.0;
        int64_t best_p_area = std::numeric_limits<int64_t>::max();

        for (size_t pi = 0; pi < np; ++pi) {
            const double ioa = rectIoA(tb, panel_boxes[pi]);
            if (ioa < 0.25) continue;
            const int64_t pa = panel_boxes[pi].area();
            if (ioa > best_ioa + 0.05 || (std::abs(ioa - best_ioa) <= 0.05 && pa < best_p_area)) {
                best_ioa = ioa;
                best_p_area = pa;
                best_p = static_cast<int>(pi);
            }
        }

        if (best_p < 0) {
            // Check >= 50% vertical alignment beside nearest panel
            const int th = std::max(1, tb.height());
            int best_dist = std::numeric_limits<int>::max();
            for (size_t pi = 0; pi < np; ++pi) {
                const RectI& pb = panel_boxes[pi];
                const int v_ov = std::max(0, std::min(tb.y2, pb.y2) - std::max(tb.y1, pb.y1));
                if (static_cast<double>(v_ov) / static_cast<double>(th) >= 0.50) {
                    int h_dist = 0;
                    if (tb.x2 < pb.x1) h_dist = pb.x1 - tb.x2;
                    else if (tb.x1 > pb.x2) h_dist = tb.x1 - pb.x2;
                    if (h_dist < best_dist) {
                        best_dist = h_dist;
                        best_p = static_cast<int>(pi);
                    }
                }
            }
        }

        if (best_p >= 0) {
            panel_texts[static_cast<size_t>(best_p)].push_back(ti);
        } else {
            unassigned_texts.push_back(ti);
        }
    }

    // 3. Recursively sort panels (with nested children) and text blocks inside each panel.
    std::vector<size_t> final_order;
    final_order.reserve(nt);

    auto emitPanelRecursive = [&](auto&& self, size_t pi) -> void {
        // Combine direct text blocks and child panels of `pi` into unified box items
        // so nested panels are ordered seamlessly with sibling text blocks inside `pi`.
        std::vector<BoxItem> local_items;
        // Encoding: idx < nt => text block `idx`; idx >= nt => child panel `idx - nt`.
        for (size_t ti : panel_texts[pi]) {
            const auto& tb = text_boxes[ti];
            local_items.push_back(BoxItem{
                ti,
                static_cast<float>(tb.x1), static_cast<float>(tb.y1),
                static_cast<float>(tb.x2), static_cast<float>(tb.y2)
            });
        }
        for (size_t child_p : panel_children[pi]) {
            const auto& pb = panel_boxes[child_p];
            local_items.push_back(BoxItem{
                nt + child_p,
                static_cast<float>(pb.x1), static_cast<float>(pb.y1),
                static_cast<float>(pb.x2), static_cast<float>(pb.y2)
            });
        }

        const auto local_order = sortBoxesGuillotineOrTournament(local_items, rtl);
        for (size_t enc : local_order) {
            if (enc < nt) {
                final_order.push_back(enc);
            } else {
                self(self, enc - nt);
            }
        }
    };

    // Top-level items: all root panels + unassigned text blocks.
    std::vector<BoxItem> root_items;
    for (size_t pi = 0; pi < np; ++pi) {
        if (panel_parent[pi] < 0) {
            const auto& pb = panel_boxes[pi];
            root_items.push_back(BoxItem{
                nt + pi,
                static_cast<float>(pb.x1), static_cast<float>(pb.y1),
                static_cast<float>(pb.x2), static_cast<float>(pb.y2)
            });
        }
    }
    for (size_t ti : unassigned_texts) {
        const auto& tb = text_boxes[ti];
        root_items.push_back(BoxItem{
            ti,
            static_cast<float>(tb.x1), static_cast<float>(tb.y1),
            static_cast<float>(tb.x2), static_cast<float>(tb.y2)
        });
    }

    const auto root_order = sortBoxesGuillotineOrTournament(root_items, rtl);
    for (size_t enc : root_order) {
        if (enc < nt) {
            final_order.push_back(enc);
        } else {
            emitPanelRecursive(emitPanelRecursive, enc - nt);
        }
    }

    return final_order;
}

void assignSpeakerClusters(
    std::vector<BubbleDetection>& detections,
    const std::vector<PanelDetection>& panels) {
    clusterSpeakersByTailRays(detections, panels);
}

}  // namespace vellum::detector
