#pragma once
#include "params.h"
#include <string>

class ConfigLoader {
public:
  static SimParams load(const std::string &filename);
  static void validate(const SimParams &params);
};
