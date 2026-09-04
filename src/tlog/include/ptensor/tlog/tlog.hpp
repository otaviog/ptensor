#pragma once

#include <string>

namespace p10 {
class Tensor;
}  // namespace p10

namespace p10::tlog {

/// Sends `tensor` to the tensor log server under the name `entry`.
///
/// The first call opens the connection, to `PTENSOR_TLOG_ADDRESS` when that
/// variable is set and to "localhost:4449" otherwise. Failures are logged, not
/// reported: logging is a debugging aid and must not change the caller's flow.
void log(const std::string& entry, const Tensor& tensor);

}  // namespace p10::tlog
