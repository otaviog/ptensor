#pragma once

#include <ptensor/dtype.hpp>

class JsDtype {
  public:
    JsDtype() = default;

    explicit JsDtype(const p10::Dtype& dtype) : dtype_(dtype) {}
    
    p10::Dtype to_p10_dtype() const {
        return dtype_;
    }

  private:
    p10::Dtype dtype_;
};