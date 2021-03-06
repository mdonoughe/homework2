/*\
 * Homework 2 - a neural network with a real-time preview
 * Copyright (C) 2010 Matthew Donoughe
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRENTY; without even the implied warrenty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. Tee the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Pubilc License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
\*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "main.h"
#include "shaderbuilder.h"

#ifdef __WIN32__
static int asprintf(char **ret, const char *format, ...) {
    va_list args;
    int retsize;
    va_start(args, format);
    retsize = _vscprintf(format, args);
    *ret = (char *)malloc((retsize + 1) * sizeof(char));
    vsprintf_s(*ret, retsize + 1, format, args);
    return retsize;
}
#endif

static const char *htanActivation = "\
        float activate(float x) {\n\
          x = clamp(x, -42.0, 42.0);\n\
          float powe = exp(2 * x);\n\
          return (powe - 1.0) / (powe + 1.0);\n\
        }\
";

static const char *logisticActivation = "\
        float activate(float x) {\n\
          x = clamp(x, -84.0, 84.0);\n\
          return 2.0 / (1.0 + exp(-x)) - 1.0;\n\
        }\
";

static const char *linearActivation = "\
        float activate(float x) {\n\
          return x;\n\
        }\
";

static const char *stepActivation = "\
        float activate(float x) {\n\
          return step(0.0, x) * 2.0 - 1.0;\n\
        }\
";

static const char *gaussian = "\
        float rbf(vec2 x, vec2 mean, float var) {\n\
          return exp(-(pow(x.x - mean.x, 2) + pow(x.y - mean.y, 2)) / (2 * var));\n\
        }\
";

static const char *fragmentShader = "\
        #version 120\n\
        #extension GL_EXT_bindable_uniform : require\n\
        uniform vec2 msStep;\n\
        uniform vec2 msHalfStep;\n\
        bindable uniform float weights[%d];\n\
        varying vec2 ftexcoord;\n\
        \n\
%s\n\
        \n\
%s\n\
        \n\
        vec4 sample(vec2 pos) {\n\
          float v00000000 = pos.x;\n\
          float v00000001 = pos.y;\n\
%s\n\
          float class1 = step(val, -0.8);\n\
          float class2 = step(0.8, val);\n\
          float unclass = 1.0 - class1 - class2;\n\
          return class1 * vec4(0.25, 0.25, 0.25, 1.0) + class2 * vec4(0.0, 0.0, 0.75, 1.0) + unclass * vec4(0.75, 0.0, 0.0, 1.0);\n\
        }\n\
        \n\
        void main(void) {\n\
          vec2 steppedBack = ftexcoord - msHalfStep;\n\
          gl_FragColor = (sample(steppedBack) + sample(vec2(steppedBack.x + msStep.x, steppedBack.y)) + sample(vec2(steppedBack.x, steppedBack.y + msStep.y)) + sample(steppedBack + msStep)) / 4.0;\n\
        }\n\
";

typedef struct Part Part;

struct Part {
  char *text;
  int len;
  Part *next;
};

void addPart(char *text, int len, Part ***eoParts) {
  Part *part = malloc(sizeof(Part));
  part->text = text;
  part->len = len;
  part->next = NULL;
  **eoParts = part;
  *eoParts = &part->next;
}

static int createEQ(char **ret) {
  int len = 0;
  int i;
  int j;
  int k;
  int lastLayerSize;
  int lastlen;
  Part *parts;
  Part **eoParts = &parts;
  int weightIndex = 0;
  int starti = 1;
  char *dest;
  char *str;
  if (nnData.isRBF) {
    starti++;
    for (j = 0; j < nnData.layerSizes[1]; j++) {
      len += lastlen = asprintf(&dest, "          float v0001%04x = rbf(vec2(v00000000, v00000001), vec2(weights[%d], weights[%d]), weights[%d]);\n", j, weightIndex, weightIndex + 1, weightIndex + 2);
      weightIndex += 3;
      addPart(dest, lastlen, &eoParts);
    }
  }
  for (i = starti; i < nnData.layers; i++) {
    for (j = 0; j < nnData.layerSizes[i]; j++) {
      if (i == nnData.layers - 1)
        len += lastlen = asprintf(&dest, "          float val = activate(weights[%d]", weightIndex++);
      else
        len += lastlen = asprintf(&dest, "          float v%04x%04x = activate(weights[%d]", i, j, weightIndex++);
      addPart(dest, lastlen, &eoParts);
      for (k = 0; k < nnData.layerSizes[i - 1]; k++) {
        char *source;
        len += lastlen = asprintf(&source, " + weights[%d] * v%04x%04x", weightIndex++, i - 1, k);
        addPart(source, lastlen, &eoParts);
      }
      len += lastlen = 3;
      addPart(strdup(");\n"), lastlen, &eoParts);
    }
    lastLayerSize = nnData.layerSizes[i];
  }
  str = (char *)malloc(len + 1);
  *ret = str;
  while(parts != NULL) {
    Part *next;
    memcpy(str, parts->text, parts->len);
    str += parts->len;
    next = parts->next;
    free(parts->text);
    free(parts);
    parts = next;
  }
  *str = 0;
  return len;
}

static const char *activation() {
  switch(nnData.function) {
    case ACTIVATION_HYPERBOLIC_TANGENT:
      return htanActivation;
    case ACTIVATION_LOGISTIC:
      return logisticActivation;
    case ACTIVATION_STEP:
      return stepActivation;
    case ACTIVATION_LINEAR:
      return linearActivation;
    default:
      return "Unknown activation function!";
  }
}

int createFShader(char **ret) {
  int retLen;
  char *eq;
  createEQ(&eq);
  retLen = asprintf(ret, fragmentShader, nnData.weightsSize, activation(), gaussian, eq);
  free(eq);
  printf("%s", *ret);
  return retLen;
}
