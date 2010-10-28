#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#ifdef TURBO
#include <pthread.h>
#endif

#include "main.h"
#include "shaderbuilder.h"
#include "nn.h"

#define PWIDTH 0.01

static const char *quadVertexShader = "\
        #version 120\n\
        attribute vec2 position;\n\
        attribute vec2 texcoord;\n\
        varying vec2 ftexcoord;\n\
        \n\
        void main(void) {\n\
          gl_Position = vec4(position, 0.0, 1.0);\n\
          ftexcoord = texcoord;\n\
        }\n\
";

static const char *pointVertexShader = "\
        #version 120\n\
        attribute vec2 position;\n\
        attribute vec3 color;\n\
        varying vec3 fcolor;\n\
        \n\
        void main(void) {\n\
          gl_Position = vec4(position, 0.0, 1.0);\n\
          fcolor = color;\n\
        }\n\
";

static const char *pointFragmentShader = "\
        #version 120\n\
        varying vec3 fcolor;\n\
        \n\
        void main(void) {\n\
          gl_FragColor = vec4(fcolor, 1.0);\n\
        }\n\
";

static const GLfloat vertices[] = {-1.0, -1.0, -10.0, -10.0, 1.0, -1.0, 10.0, -10.0, -1.0, 1.0, -10.0, 10.0, 1.0, 1.0, 10.0, 10.0};

GLData glData;
NNData nnData;

static void reshape(int width, int height) {
  glViewport(0, 0, width, height);
  glUseProgram(glData.quadProgram);
  // texcoord space size / (4 * screen space size)
  glUniform2f(glData.halfStepUniform, 5.0 / (double) width, 5.0 / (double) height);
  // texcoord space size / (2 * screen space size)
  glUniform2f(glData.stepUniform, 10.0 / (double) width, 10.0 / (double) height);
  glutPostRedisplay();
}

static void display() {
  glUseProgram(glData.quadProgram);
  glBindBuffer(GL_ARRAY_BUFFER, glData.quadVbo);
  glEnableVertexAttribArray(glData.quadPositionAttribute);
  glVertexAttribPointer(glData.quadPositionAttribute, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (GLfloat*)0);
  glEnableVertexAttribArray(glData.quadTexCoordAttribute);
  glVertexAttribPointer(glData.quadTexCoordAttribute, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (GLfloat*)0 + 2);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

  glUseProgram(glData.pointsProgram);
  glBindBuffer(GL_ARRAY_BUFFER, glData.pointsVbo);
  glEnableVertexAttribArray(glData.pointsPositionAttribute);
  glVertexAttribPointer(glData.pointsPositionAttribute, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLfloat*)0);
  glEnableVertexAttribArray(glData.pointsColorAttribute);
  glVertexAttribPointer(glData.pointsColorAttribute, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLfloat*)0 + 2);
  glDrawArrays(GL_QUADS, 0, nnData.inputsSize * 4);

  glutSwapBuffers();
}

static void checkShaderError(GLuint shader) {
  GLint glRet;
  glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &glRet);
  if (glRet != 0) {
    char *error = malloc(glRet);
    glGetShaderInfoLog(shader, glRet, NULL, error);
    fprintf(stderr, "%s\n", error);
    free(error);
  }
}

static void checkProgramError(GLuint shader) {
  GLint glRet;
  glGetProgramiv(shader, GL_INFO_LOG_LENGTH, &glRet);
  if (glRet != 0) {
    char *error = malloc(glRet);
    glGetProgramInfoLog(shader, glRet, NULL, error);
    fprintf(stderr, "%s\n", error);
    free(error);
  }
}

static void idle() {
#ifndef TURBO
  // if we aren't in turbo mode, learn once per redraw
  double mse = learn();
  printf("%d\t%f\n", nnData.epoch, mse);
#endif
  glUseProgram(glData.quadProgram);
  glUniform1fv(glData.weightsUniform, nnData.weightsSize, nnData.weights);
  glutPostRedisplay();
}

#ifdef TURBO
// in turbo mode we learn as fast as possible
// this is not thread safe, but the display just gets a little weird at worst
static void *learnStuff(void *v) {
  while(1) {
    double mse = learn();
    printf("%d\t%f\n", nnData.epoch, mse);
  }
  return NULL;
}

static void turboDisplay() {
  display();
  glutDisplayFunc(&display);
  pthread_t thread;
  pthread_create(&thread, NULL, &learnStuff, NULL);
}
#endif

int main(int argc, char **argv) {
  // create window
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
  glutInitWindowSize(512, 512);
  glutCreateWindow("Homework 2");

  // initialize network
  initNN(&argc, argv);

  {
    // build quad vertex shader
    GLuint vshader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vshader, 1, &quadVertexShader, NULL);
    glCompileShader(vshader);
    checkShaderError(vshader);

    // build quad fragment shader
    GLuint fshader = glCreateShader(GL_FRAGMENT_SHADER);
    char *fshaderSource;
    int fshaderSourceLen = createFShader(&fshaderSource);
    glShaderSource(fshader, 1, (const char **)&fshaderSource, &fshaderSourceLen);
    glCompileShader(fshader);
    free(fshaderSource);
    checkShaderError(fshader);

    // link quad program
    glData.quadProgram = glCreateProgram();
    glAttachShader(glData.quadProgram, vshader);
    glAttachShader(glData.quadProgram, fshader);
    glLinkProgram(glData.quadProgram);
    checkProgramError(glData.quadProgram);
    glUseProgram(glData.quadProgram);

    // find uniforms
    glData.halfStepUniform = glGetUniformLocation(glData.quadProgram, "halfStep");
    glData.stepUniform = glGetUniformLocation(glData.quadProgram, "step");
    glData.quadPositionAttribute = glGetAttribLocation(glData.quadProgram, "position");
    glData.quadTexCoordAttribute = glGetAttribLocation(glData.quadProgram, "texcoord");
    glData.weightsUniform = glGetUniformLocation(glData.quadProgram, "weights");

    // build quad VBO
    glGenBuffers(1, &glData.quadVbo);
    glBindBuffer(GL_ARRAY_BUFFER, glData.quadVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // build weights UBO
    GLint size = glGetUniformBufferSizeEXT(glData.quadProgram, glData.weightsUniform);
    glGenBuffers(1, &glData.ubo);
    glBindBuffer(GL_UNIFORM_BUFFER_EXT, glData.ubo);
    glBufferData(GL_UNIFORM_BUFFER_EXT, size, NULL, GL_STREAM_DRAW);
    glUniformBufferEXT(glData.quadProgram, glData.weightsUniform, glData.ubo);

    // build point vertex shader
    vshader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vshader, 1, &pointVertexShader, NULL);
    glCompileShader(vshader);
    checkShaderError(vshader);

    // build point fragment shader
    fshader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fshader, 1, &pointFragmentShader, NULL);
    glCompileShader(fshader);
    checkShaderError(fshader);

    // link point program
    glData.pointsProgram = glCreateProgram();
    glAttachShader(glData.pointsProgram, vshader);
    glAttachShader(glData.pointsProgram, fshader);
    glLinkProgram(glData.pointsProgram);
    checkProgramError(glData.pointsProgram);
    glUseProgram(glData.pointsProgram);

    // find uniforms
    glData.pointsPositionAttribute = glGetAttribLocation(glData.pointsProgram, "position");
    glData.pointsColorAttribute = glGetAttribLocation(glData.pointsProgram, "color");

    // build point VBO
    unsigned int pointsArraySize = nnData.inputsSize * 4 * (2 + 3) * sizeof(GLfloat);
    GLfloat *pointsArray = malloc(pointsArraySize);
    int i;
    for (i = 0; i < nnData.inputsSize; i++) {
      InputVector *iv = nnData.inputs + i;
      GLfloat *point = pointsArray + i * 4 * (2 + 3);
      GLfloat color[3] = {0.0, 0.0, 0.0};
      if (iv->target == 1)
        color[2] = 1.0;
      point[0] = iv->x / 10.0 - PWIDTH;
      point[1] = iv->y / 10.0 - PWIDTH;
      memcpy(point + 2, color, sizeof(color));
      point[5] = iv->x / 10.0 + PWIDTH;
      point[6] = iv->y / 10.0 - PWIDTH;
      memcpy(point + 7, color, sizeof(color));
      point[10] = iv->x / 10.0 + PWIDTH;
      point[11] = iv->y / 10.0 + PWIDTH;
      memcpy(point + 12, color, sizeof(color));
      point[15] = iv->x / 10.0 - PWIDTH;
      point[16] = iv->y / 10.0 + PWIDTH;
      memcpy(point + 17, color, sizeof(color));
    }
    glGenBuffers(1, &glData.pointsVbo);
    glBindBuffer(GL_ARRAY_BUFFER, glData.pointsVbo);
    glBufferData(GL_ARRAY_BUFFER, pointsArraySize, pointsArray, GL_STATIC_DRAW);
    free(pointsArray);
  }

  glutReshapeFunc(&reshape);
#ifdef TURBO
  glutDisplayFunc(&turboDisplay);
#else
  glutDisplayFunc(&display);
#endif
  glutIdleFunc(&idle);
  glutReportErrors();
  glutMainLoop();
  return 0;
}
