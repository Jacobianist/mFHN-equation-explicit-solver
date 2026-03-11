#pragma once
#include <string>

#include "params.h"

class ConfigLoader {
   public:
    static SimParams load(const std::string& filename);
    static void validate(const SimParams& params);
};
