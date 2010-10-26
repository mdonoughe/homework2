#include <OpenGL/gl.h>

typedef enum ActivationFunction {
  ACTIVATION_HYPERBOLIC_TANGENT,
  ACTIVATION_LOGISTIC
} ActivationFunction;

typedef struct GLData {
  GLuint quadVbo;
  GLuint ubo;
  GLuint quadProgram;
  GLuint quadPositionAttribute;
  GLuint quadTexCoordAttribute;
  GLuint pointsVbo;
  GLuint pointsProgram;
  GLuint pointsPositionAttribute;
  GLuint pointsColorAttribute;

  // a quarter pixel(when doing 4x multisampling)
  GLuint halfStepUniform;
  // a half pixel(when doing 4x multisampling)
  GLuint stepUniform;
  // where to send the weight values
  GLuint weightsUniform;
} GLData;

typedef struct InputVector {
  float x;
  float y;
  int target;
} InputVector;

typedef struct NNData {
  // number of weights
  unsigned int weightsSize;
  // all weights arranged as follows
  // layer 0: (no weights because it's the input layer)
  // layer 1: (if it has two nodes)
  // bias, x, y
  // bias, x, y
  // layer 2: (if it has one node)
  // bias, layer1-0, layer1-1
  // GLfloat because this is shipped into OpenGL
  GLfloat *weights;
  // number of values(one for every node, and a bias for every layer)
  unsigned int valuesSize;
  // number of preactivation values(one for every node)
  unsigned int preActivatesSize;
  // all values arranged as follows
  // layer 0:
  // bias, x, y
  // layer 1:
  // bias, layer1-0, layer1-1
  // layer 2:
  // bias, layer2-0
  double *values;
  // all preactivates arranged as follows
  // layer 0: (no preactivations because these values are not computed)
  // layer 1:
  // layer1-0, layer1-1
  // layer 2:
  // layer2-0
  double *preActivates;
  // number of errors
  unsigned int errorsSize;
  // all errors arranged as follows
  // layer 0: (no errors because these values are not computed)
  // layer 1:
  // layer1-0, layer1-1
  // layer 2:
  // layer2-0
  double *errors;
  // all momentums arranged as follows
  // layer 0: (no weights because it's the input layer)
  // layer 1: (if it has two nodes)
  // bias, x, y
  // bias, x, y
  // layer 2: (if it has one node)
  // bias, layer1-0, layer1-1
  double *momentums;
  // number of layers
  unsigned int layers;
  // size of each layer
  unsigned int *layerSizes;

  // just pointers to the layer offsets in the corresponding arrays
  GLfloat **layerWeights;
  double **layerPreActivates;
  double **layerValues;
  double **layerErrors;
  double **layerMomentums;

  // number of input vectors
  unsigned int inputsSize;
  // all input vectors
  InputVector *inputs;
  // a randomized list of input vectors
  InputVector **shuffledInputs;

  // factors
  double learnRate;
  double momentum;

  // used to select the activation function for the shader
  ActivationFunction function;

  // pointers to the functions called by the C code
  double (*activate)(double);
  double (*derive)(double);

  // the number of times learn has been called
  unsigned int epoch;
} NNData;

extern GLData glData;
extern NNData nnData;
