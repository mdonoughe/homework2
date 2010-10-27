#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "nn.h"
#include "main.h"

typedef struct Input Input;

struct Input {
  InputVector input;
  Input *next;
};

static void addInput(double x, double y, int target, Input ***eoInputs) {
  Input *input = malloc(sizeof(Input));
  input->input.x = x;
  input->input.y = y;
  input->input.target = 2 * target - 1;
  input->next = NULL;
  **eoInputs = input;
  *eoInputs = &input->next;
}

static void shuffle() {
  int i;
  for (i = 0; i < nnData.inputsSize - 1; i++) {
    // pick a number between i and the end of the list
    int offset = i + random() % (nnData.inputsSize - i);
    // swap the current index and the random one
    InputVector *iv = nnData.shuffledInputs[i];
    nnData.shuffledInputs[i] = nnData.shuffledInputs[offset];
    nnData.shuffledInputs[offset] = iv;
  }
}

static void readTrainingSet(char *file) {
  // open file
  FILE *f = fopen(file, "r");
  double x;
  double y;
  int target;
  Input *inputs;
  Input **eoInputs = &inputs;
  int count = 0;
  // read inputs into linked list
  while (fscanf(f, "%lf\t%lf\t%d\n", &x, &y, &target) == 3) {
    addInput(x, y, target, &eoInputs);
    count++;
  }
  fclose(f);
  // turn linked list into array
  nnData.inputs = malloc(sizeof(InputVector) * count);
  nnData.inputsSize = count;
  InputVector *output = nnData.inputs;
  while (inputs != NULL) {
    memcpy(output, inputs, sizeof(InputVector));
    output++;
    Input *next = inputs->next;
    free(inputs);
    inputs = next;
  }
  // copy the list into the shuffled list
  // this is in order but will be shuffled on the first learn
  nnData.shuffledInputs = malloc(sizeof(InputVector *) * count);
  int i;
  for (i = 0; i < count; i++) {
    nnData.shuffledInputs[i] = nnData.inputs + i;
  }
}

static double activate_htan(double x) {
  // 42: the answer to life, the universe, and when to stop raising e to 2 * x
  if (x > 42.0)
    return 1.0;
  if (x < -42.0)
    return -1.0;
  double powe = exp(2.0 * x);
  return (powe - 1.0) / (powe + 1.0);
}

static double activate_log(double x) {
  if (x > 84.0)
    return 1.0;
  if (x < -84.0)
    return -1.0;
  return 2.0 / (1.0 + exp(-x)) - 1.0;
}

static double activate_step(double x) {
  if (x < 0.0)
    return -1.0;
  return 1.0;
}

static double derive_htan(double x) {
  if (x > 84.0 || x < -84.0)
    return 0.0;
  // 1 / cosh(x)**2
  // 1 / (0.5 * (e ** x + e ** -x)) ** 2
  // 4 / (e ** x + e ** -x) ** 2
  return 4.0 * pow(exp(x) + exp(-x), -2);
}

static double derive_log(double x) {
  if (x > 84.0 || x < -84.0)
    return 0.0;
  return 2.0 / ((1.0 + exp(-x)) * (1.0 + exp(x)));
}

static double derive_step(double x) {
  return 1.0;
}

double learn() {
  double mse = 0.0;
  int i, j, k, l;
#ifdef SLOW
  i = nnData.epoch % nnData.inputsSize;
  {
#else
  for (i = 0; i < nnData.inputsSize; i++) {
#endif
    if (i == 0)
      shuffle();
    // clear values
    bzero(nnData.preActivates, nnData.preActivatesSize * sizeof(double));
    // clear errors
    bzero(nnData.errors, nnData.errorsSize * sizeof(double));
    // input values
    nnData.values[1] = nnData.shuffledInputs[i]->x;
    nnData.values[2] = nnData.shuffledInputs[i]->y;

    unsigned int startLayer = 1;
    if (nnData.isRBF) {
      startLayer++;
      int stride = nnData.layerSizes[0] + 1;
      for (k = 0; k < nnData.layerSizes[1]; k++) {
        nnData.layerValues[1][k + 1] = exp(-(pow(nnData.layerValues[0][1] - nnData.weights[stride * k], 2) + pow(nnData.layerValues[0][2] - nnData.weights[stride * k + 1], 2)) / (2 * nnData.weights[stride * k + 2]));
      }
    }

    // find outputs - forward
    // myWeights points to the weights for the current set of inputs and output
    GLfloat *myWeights = nnData.layerWeights[startLayer - 1];
    for (j = startLayer; j < nnData.layers; j++) {
      for (k = 0; k < nnData.layerSizes[j]; k++) {
        for (l = 0; l < nnData.layerSizes[j - 1] + 1; l++) {
          nnData.layerPreActivates[j][k] += myWeights[l] * nnData.layerValues[j - 1][l];
        }
        // advance myWeights to the next node
        myWeights += nnData.layerSizes[j - 1] + 1;
        // activate!
        nnData.layerValues[j][k + 1] = nnData.activate(nnData.layerPreActivates[j][k]);
      }
    }

    // find the errors - backward
    nnData.errors[nnData.errorsSize - 1] = nnData.shuffledInputs[i]->target - nnData.values[nnData.valuesSize - 1];
    for (j = nnData.layers - 2; j >= startLayer; j--) {
      // myWeights here goes something like 6 7 3 4 5 1 2
      // jump backwards here
      myWeights = nnData.layerWeights[j];
      for (k = 0; k < nnData.layerSizes[j + 1]; k++) {
        for (l = 0; l < nnData.layerSizes[j]; l++) {
          // sum up errors
          nnData.layerErrors[j][l] += myWeights[l + 1] * nnData.layerErrors[j + 1][k];
        }
        // advance myWeights
        myWeights += nnData.layerSizes[j] + 1;
      }
    }

    // learn - forward
    // myMomentums and myWeights are the same idea
    myWeights = nnData.layerWeights[startLayer - 1];
    double *myMomentums = nnData.layerMomentums[startLayer - 1];
    for (j = startLayer; j < nnData.layers; j++) {
      for (k = 0; k < nnData.layerSizes[j]; k++) {
        double delta = nnData.derive(nnData.layerPreActivates[j][k]) * nnData.layerErrors[j][k] * nnData.learnRate;
        for (l = 0; l < nnData.layerSizes[j - 1] + 1; l++) {
          // calculate the change into the momentum term
          myMomentums[l] = delta * nnData.layerValues[j - 1][l] + nnData.momentum * myMomentums[l];
          // update the weight
          myWeights[l] += myMomentums[l];
        }
        // advance pointers
        myWeights += nnData.layerSizes[j - 1] + 1;
        myMomentums += nnData.layerSizes[j - 1] + 1;
      }
    }
    // add error to total
    mse += 0.5 * pow(nnData.errors[nnData.errorsSize - 1], 2);
#if 0
    // print tons of information
    // will slow things down a lot
    printf("values:\n");
    for (j = 0; j < nnData.valuesSize; j++) {
      printf("%f\n", nnData.values[j]);
    }
    printf("errors:\n");
    for (j = 0; j < nnData.errorsSize; j++) {
      printf("%f\n", nnData.errors[j]);
    }
    printf("momentums:\n");
    for (j = 0; j < nnData.weightsSize; j++) {
      printf("%f\n", nnData.momentums[j]);
    }
    printf("weights:\n");
    for (j = 0; j < nnData.weightsSize; j++) {
      printf("%f\n", nnData.weights[j]);
    }
#endif
  }
  nnData.epoch++;
#ifdef SLOW
  return mse;
#else
  return mse / nnData.inputsSize;
#endif
}

static void cluster() {
  shuffle();
  int i;
  int j;
  int stride = nnData.layerSizes[0] + 1;
  int *assignments = malloc(nnData.inputsSize * sizeof(int));
  int *oldAssignments = malloc(nnData.inputsSize * sizeof(int));
  bzero(oldAssignments, nnData.inputsSize * sizeof(int));
  double *newMeans = malloc(nnData.layerSizes[0] * nnData.layerSizes[1] * sizeof(double));
  unsigned int *assigned = malloc(nnData.layerSizes[1] * sizeof(unsigned int));
  char done = 0;
  // randomly select starting points
  for (i = 0; i < nnData.layerSizes[1]; i++) {
    nnData.weights[i * stride] = nnData.shuffledInputs[i]->x;
    nnData.weights[i * stride + 1] = nnData.shuffledInputs[i]->y;
    // init variance
    nnData.weights[i * stride + 2] = 0.0;
  }
  // organize into clusters
  while (!done) {
    done = 1;
    bzero(newMeans, nnData.layerSizes[0] * nnData.layerSizes[1] * sizeof(double));
    bzero(assigned, nnData.layerSizes[1] * sizeof(unsigned int));
    bzero(assignments, nnData.inputsSize * sizeof(int));
    for (i = 0; i < nnData.inputsSize; i++) {
      double distance = hypot(nnData.inputs[i].x - nnData.weights[0], nnData.inputs[i].y - nnData.weights[1]);
      for (j = 1; j < nnData.layerSizes[1]; j++) {
        double ndistance = hypot(nnData.inputs[i].x - nnData.weights[j * stride], nnData.inputs[i].y - nnData.weights[j * stride + 1]);
        if (ndistance < distance) {
          assignments[i] = j;
          distance = ndistance;
        }
      }
      done &= oldAssignments[i] == assignments[i];
      newMeans[nnData.layerSizes[0] * assignments[i]] += nnData.inputs[i].x;
      newMeans[nnData.layerSizes[0] * assignments[i] + 1] += nnData.inputs[i].y;
      assigned[assignments[i]]++;
    }
    for (i = 0; i < nnData.layerSizes[1]; i++) {
      for (j = 0; j < nnData.layerSizes[0]; j++) {
        nnData.weights[i * stride + j] = newMeans[i * nnData.layerSizes[0] + j] / assigned[i];
      }
    }
    if (!done) {
      memcpy(oldAssignments, assignments, nnData.inputsSize * sizeof(int));
    }
  }
  // find variance
  for (i = 0; i < nnData.inputsSize; i++) {
    nnData.weights[assignments[i] * stride + 2] += pow(nnData.inputs[i].x - nnData.weights[assignments[i] * stride], 2) + pow(nnData.inputs[i].y - nnData.weights[assignments[i] * stride + 1], 2);
  }
  for (i = 0; i < nnData.layerSizes[1]; i++) {
    nnData.weights[i * stride + 2] /= assigned[i];
    nnData.weights[i * stride + 2] += 0.01;
  }
  // free memory
  free(newMeans);
  free(oldAssignments);
  free(assignments);
  free(assigned);
}

void initNN(int *argc, char **argv) {
  // read data file
  if (*argc > 2 && strcmp(argv[1], "-f") == 0) {
    // we got a file from the command line
    readTrainingSet(argv[2]);
    // remove this part of the command line so it isn't processed later
    memmove(argv + 1, argv + 3, (*argc - 3) * sizeof(char *));
    (*argc) -= 2;
  } else {
    readTrainingSet("spiral.dat");
  }

  nnData.momentum = 0.95;
  nnData.learnRate = 0.0001;
  nnData.epoch = 0;
  nnData.isRBF = 1;

  // the number of hidden layers is specified on the command line
  // main 5 5 produces a network 2 5 5 1
  nnData.layers = *argc + 1;
  nnData.layerSizes = malloc(nnData.layers * sizeof(unsigned int));
  nnData.layerSizes[0] = 2;
  nnData.weightsSize = 0;
  nnData.valuesSize = 5; // this 5 is the final value, the inputs, and 2 biases
  nnData.preActivatesSize = 1; // this 1 is the final value
  nnData.errorsSize = 1; // the final value
  int i;
  // populate layerSizes and determine the sizes of arrays we need to allocate
  for (i = 1; i < *argc; i++) {
    nnData.layerSizes[i] = atoi(argv[i]);
    // each node has a weight for each source node and a bias
    nnData.weightsSize += (1 + nnData.layerSizes[i - 1]) * nnData.layerSizes[i];
    // each node has one value, and each layer has a bias
    // the output layer has an unused bias value in it
    nnData.valuesSize += 1 + nnData.layerSizes[i];
    // one preactivate and error space per node
    nnData.preActivatesSize += nnData.layerSizes[i];
    nnData.errorsSize += nnData.layerSizes[i];
  }
  // the output layer is not handled by the loop
  nnData.layerSizes[i] = 1;
  nnData.weightsSize += (1 + nnData.layerSizes[i - 1]) * 1;
  // these are part of the second layer, but extra RBF stuff
  if (nnData.isRBF) {
    // we'll let the input layer keep its bias, even though it's unused
    // no preactivates or errors for the RBF layer
    nnData.preActivatesSize -= nnData.layerSizes[1];
    nnData.errorsSize -= nnData.layerSizes[1];
  }

  // set argc to 1 because we consumed all the arguments
  *argc = 1;

  // setting nnData.function changes the function used by the shader too!
  nnData.function = ACTIVATION_HYPERBOLIC_TANGENT;
  //nnData.function = ACTIVATION_LOGISTIC;
  // we don't want to use activation for a single perceptron
  if (nnData.layers - (nnData.isRBF ? 1 : 0) == 2)
    nnData.function = ACTIVATION_STEP;
  // but for the C code we'll just use a virtual function call to save cycles
  switch (nnData.function) {
    case ACTIVATION_HYPERBOLIC_TANGENT:
      nnData.activate = &activate_htan;
      nnData.derive = &derive_htan;
      break;
    case ACTIVATION_LOGISTIC:
      nnData.activate = &activate_log;
      nnData.derive = &derive_log;
      break;
    case ACTIVATION_STEP:
      nnData.activate = &activate_step;
      nnData.derive = &derive_step;
  }

  // allocate our arrays
  nnData.weights = malloc(nnData.weightsSize * sizeof(GLfloat));
  nnData.momentums = malloc(nnData.weightsSize * sizeof(double));
  nnData.values = malloc(nnData.valuesSize * sizeof(double));
  nnData.preActivates = malloc(nnData.preActivatesSize * sizeof(double));
  nnData.errors = malloc(nnData.errorsSize * sizeof(double));

  // allocate convenience arrays
  nnData.layerWeights = malloc(nnData.layers * sizeof(GLfloat *));
  nnData.layerMomentums = malloc(nnData.layers * sizeof(double *));
  nnData.layerPreActivates = malloc(nnData.layers * sizeof(double *));
  nnData.layerValues = malloc(nnData.layers * sizeof(double *));
  nnData.layerErrors = malloc(nnData.layers * sizeof(double *));
  // initialize convenience arrays
  nnData.layerWeights[0] = nnData.weights;
  nnData.layerMomentums[0] = nnData.momentums;
  nnData.layerValues[0] = nnData.values;
  // these two are bogus values because the input layer does not have them
  // they need populated because the other layers are offsets from the first
  nnData.layerPreActivates[0] = nnData.preActivates - nnData.layerSizes[0];
  nnData.layerErrors[0] = nnData.errors - nnData.layerSizes[0];
  if (nnData.isRBF) {
    nnData.layerPreActivates[0] -= nnData.layerSizes[1];
    nnData.layerErrors[0] -= nnData.layerSizes[1];
  }
  for (i = 1; i < nnData.layers; i++) {
    nnData.layerWeights[i] = nnData.layerWeights[i - 1] + (nnData.layerSizes[i - 1] + 1) * nnData.layerSizes[i];
    nnData.layerMomentums[i] = nnData.layerMomentums[i - 1] + (nnData.layerSizes[i - 1] + 1) * nnData.layerSizes[i];
    nnData.layerValues[i] = nnData.layerValues[i - 1] + nnData.layerSizes[i - 1] + 1;
    nnData.layerErrors[i] = nnData.layerErrors[i - 1] + nnData.layerSizes[i - 1];
    nnData.layerPreActivates[i] = nnData.layerPreActivates[i - 1] + nnData.layerSizes[i - 1];
  }
  // change the bogus values to nulls
  nnData.layerPreActivates[0] = NULL;
  nnData.layerErrors[0] = NULL;
  nnData.layerWeights[nnData.layers - 1] = NULL;
  if (nnData.isRBF) {
    nnData.layerPreActivates[1] = NULL;
    nnData.layerErrors[1] = NULL;
  }

  // initialize weights
  for (i = 0; i < nnData.weightsSize; i++) {
    nnData.weights[i] = 0.5 - random() / (double) 0x7fffffff;
  }
  // initialize momentums
  bzero(nnData.momentums, nnData.weightsSize * sizeof(double));
  // initialize biases
  for (i = 0; i < nnData.layers; i++) {
    nnData.layerValues[i][0] = 1.0;
  }
  if (nnData.isRBF) {
    cluster();
  }
}
