#pragma once

#include "../app_context.hpp"

namespace mangaflow::routes {

void register_system_routes(httplib::Server& server, AppContext& ctx);

}  // namespace mangaflow::routes
