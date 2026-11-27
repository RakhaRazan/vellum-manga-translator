#pragma once

#include "../app_context.hpp"

namespace mangaflow::routes {

void register_export_routes(httplib::Server& server, AppContext& ctx);

}  // namespace mangaflow::routes
