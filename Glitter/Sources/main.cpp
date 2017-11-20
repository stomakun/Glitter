// Local Headers
#include "glitter.hpp"

// System Headers
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// Standard Headers
#include <cstdio>
#include <cstdlib>

#include <cassert>

#include <memory>
#include <vector>
#include <iostream>

namespace gl {

inline const char *GLGetErrorString(GLenum error) {
  switch (error) {
    case GL_NO_ERROR:
      return "GL_NO_ERROR";
    case GL_INVALID_ENUM:
      return "GL_INVALID_ENUM";
    case GL_INVALID_VALUE:
      return "GL_INVALID_VALUE";
    case GL_INVALID_OPERATION:
      return "GL_INVALID_OPERATION";
    case GL_STACK_OVERFLOW:
      return "GL_STACK_OVERFLOW";
    case GL_STACK_UNDERFLOW:
      return "GL_STACK_UNDERFLOW";
    case GL_OUT_OF_MEMORY:
      return "GL_OUT_OF_MEMORY";
    default:
      return "Unknown OpenGL error code";
  }
}

}  // namespace gl

void OPENGL_ABSORB_ERROS() {
  while (glGetError() != GL_NO_ERROR)
    ;
}

// TODO(zhixunt): When porting to TVM, change this to
//   CHECK(err == GL_NO_ERROR) << ...;
void OPENGL_CHECK_ERROR() {
  GLenum err = glGetError();
  if (err != GL_NO_ERROR) {
    std::cerr << "OpenGL error, code=" << err << ": "
              << gl::GLGetErrorString(err) << std::endl;
    assert(false);
  }
}

/*!
 * \brief Protected OpenGL call.
 * \param func Expression to call.
 */
#define OPENGL_CALL(func)                                                      \
  {                                                                            \
    (func);                                                                    \
    OPENGL_CHECK_ERROR();                                                      \
  }

void GlfwErrorCallback(int err, const char *str) {
  std::cerr << "Error: [" << err << "] " << str << std::endl;
}

// Don't need to change this.
// We want to draw 2 giant triangles that cover the whole screen.
struct Vertex {
  float x, y;
};
Vertex vertices[] = {
    {-1.f, -1.f},
    {1.0f, -1.f},
    {1.0f, 1.0f},
    {-1.f, -1.f},
    {-1.f, 1.0f},
    {1.0f, 1.0f},
};

// Don't need to change this.
// The vertex shader only needs to take in the triangle points.
// No need for point transformations.
static const char *vertex_shader_text =
    "#version 330 core\n"
        "in vec2 point; // input to vertex shader\n"
        "void main() {\n"
        "  gl_Position = vec4(point, 0.0, 1.0);\n"
        "}\n";

// This is the main part.
static const char *fragment_shader_text =
    "#version 330 core\n"
        "uniform int width;\n"
        "uniform int height;\n"
        "uniform sampler1D texture0;\n"
        "uniform sampler1D texture1;\n"
        "out vec4 color;\n"
        "void main() {\n"
        "  // TODO(zhixunt): Calculate pixel index.\n"
        "  // gl_FragColor = vec4(gl_FragCoord.x / float(width), gl_FragCoord.y / float(height), 0.75, 1.0)\n;"
        "  color = vec4(\n"
        "    texture(texture0, 0.0).r + texture(texture1, 0.0).r,\n"
        "    0.0,\n"
        "    0.0,\n"
        "    1.0\n"
        "  );\n"
        "}\n";

// The following functions don't need to be understood.
// They are standard practices.
GLuint CreateShader(GLenum shader_kind, const char *shader_src);
GLuint CreateProgram(GLuint vertex_shader, GLuint fragment_shader);
GLuint CreateProgram(const char *vertex_shader_src, const char *fragment_shader_src);

int main(int argc, char *argv[]) {
  std::cout << "Hello, World!!" << std::endl;

  // Set an error handler.
  // This can be called before glfwInit().
  glfwSetErrorCallback(&GlfwErrorCallback);

  // Initialize GLFW.
  if (glfwInit() != GL_TRUE) {
    std::cout << "glfwInit() failed!" << std::endl;
    return 1;
  }

  GLint width = 640;
  GLint height = 480;

  // Create a window.
  // TODO(zhixunt): GLFW allows us to create an invisible window.
  // TODO(zhixunt): On retina display, window size is different from framebuffer size.
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  GLFWwindow *window = glfwCreateWindow(width, height, "My Title", nullptr, nullptr);
  if (window == nullptr) {
    std::cout << "glfwCreateWindow() failed!" << std::endl;
    return 1;
  }

  std::cout
      << "OpenGL version: "
      << glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MAJOR)
      << "."
      << glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MINOR)
      << "."
      << glfwGetWindowAttrib(window, GLFW_CONTEXT_REVISION)
      << std::endl;

  // Before using any OpenGL API, we must specify a context.
  glfwMakeContextCurrent(window);

  gladLoadGL();
  fprintf(stderr, "OpenGL %s\n", glGetString(GL_VERSION));

  OPENGL_CHECK_ERROR();

  GLuint program = CreateProgram(vertex_shader_text, fragment_shader_text);

  OPENGL_CALL(glUseProgram(program));

  GLuint vertex_buffer;
  OPENGL_CALL(glGenBuffers(1, &vertex_buffer));
  OPENGL_CALL(glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer));
  OPENGL_CALL(glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW));

  GLuint vaoId = 0;
  glGenVertexArrays(1, &vaoId);
  glBindVertexArray(vaoId);

  auto point_attrib = static_cast<GLuint>(glGetAttribLocation(program, "point"));
  std::cout << "Attrib Location" << point_attrib << std::endl;
  OPENGL_CALL(glEnableVertexAttribArray(point_attrib));
  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
  OPENGL_CALL(glVertexAttribPointer(point_attrib, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), nullptr));

  glfwGetFramebufferSize(window, &width, &height);
  auto width_uniform = glGetUniformLocation(program, "width");
  auto height_uniform = glGetUniformLocation(program, "height");
  OPENGL_CALL(glUniform1i(width_uniform, width));
  OPENGL_CALL(glUniform1i(height_uniform, height));
  OPENGL_CALL(glViewport(0, 0, width, height));

  // Set up the first texture.
  // https://www.opengl.org/discussion_boards/showthread.php/174926-when-to-use-glActiveTexture
  // Consider the internal OpenGL texture system as this.
  //   struct TextureUnit {
  //     GLuint target_texture_1D;
  //     GLuint target_texture_2D;
  //     GLuint target_texture_3D;
  //     GLuint target_texture_cube;
  //     ...
  //   };
  //   TextureUnit texture_units[GL_MAX_TEXTURE_IMAGE_UNITS];
  //   GLuint curr_texture_unit; // global state!!!
  //
  // Then:
  //   "glActiveTexture(GL_TEXTURE0);"
  //     <=>
  //   "curr_texture_unit = 0;"
  //
  //   "glBindTexture(GL_TEXTURE_1D, texture0);"
  //     <=>
  //   "texture_units[curr_texture_unit].target_texture_1D = texture0;"
  //
  {
    GLuint texture0;
    GLsizei texture0_width = 100;
    GLfloat texture0_data[100] = {0.5f};

    // Create a texture.
    OPENGL_CALL(glGenTextures(1, &texture0));

    // See comments above.
    OPENGL_CALL(glActiveTexture(GL_TEXTURE0));
    OPENGL_CALL(glBindTexture(GL_TEXTURE_1D, texture0));

    // Similar to cudaMemcpy.
    OPENGL_CALL(glTexImage1D(GL_TEXTURE_1D, 0, GL_RED, texture0_width, 0, GL_RED, GL_FLOAT, texture0_data));

    // Bind uniform "texture0" to GL_TEXTURE0.
    GLint texture0_uniform = glGetUniformLocation(program, "texture0");
    OPENGL_CALL(glUniform1i(texture0_uniform, 0));

    // TODO(zhixunt): What is this?
    OPENGL_CALL(glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    OPENGL_CALL(glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    OPENGL_CALL(glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
    OPENGL_CALL(glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
  }

  {
    GLuint texture1;
    GLsizei texture1_width = 100;
    GLfloat texture1_data[100] = {0.5f};
    OPENGL_CALL(glGenTextures(1, &texture1));
    OPENGL_CALL(glActiveTexture(GL_TEXTURE1));
    OPENGL_CALL(glBindTexture(GL_TEXTURE_1D, texture1));
    OPENGL_CALL(glTexImage1D(GL_TEXTURE_1D, 0, GL_RED, texture1_width, 0, GL_RED, GL_FLOAT, texture1_data));
    GLint texture1_uniform = glGetUniformLocation(program, "texture1");
    OPENGL_CALL(glUniform1i(texture1_uniform, 1));
    OPENGL_CALL(glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    OPENGL_CALL(glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    OPENGL_CALL(glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
    OPENGL_CALL(glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
  }

  while (glfwWindowShouldClose(window) == GL_FALSE) {
    glClear(GL_COLOR_BUFFER_BIT);
    OPENGL_CALL(glDrawArrays(GL_TRIANGLES, 0, 6));
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  {
    // Create frame buffer.
    GLuint frame_buffer = 0;
    glGenFramebuffers(1, &frame_buffer);
    glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer);

    // The texture we're going to render to
    GLuint target_texture;
    glGenTextures(1, &target_texture);

    // "Bind" the newly created texture : all future texture functions will modify this texture
    glBindTexture(GL_TEXTURE_2D, target_texture);

    // Give an empty image to OpenGL ( the last "0" )
    glTexImage2D(GL_TEXTURE_2D, /*level=*/0, GL_RGB, width, height, /*border=*/0, GL_RGB, GL_UNSIGNED_BYTE, /*pixels=*/
                 nullptr);

    // Poor filtering. Needed !
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    // The depth buffer
    GLuint depthrenderbuffer;
    glGenRenderbuffers(1, &depthrenderbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, depthrenderbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthrenderbuffer);

    // Set "renderedTexture" as our colour attachement #0
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, target_texture, 0);

    // Set the list of draw buffers.
    GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, DrawBuffers); // "1" is the size of DrawBuffers

    // Always check that our framebuffer is ok
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      return false;

    // Render to our framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer);
    glViewport(0, 0, width, height); // Render on the whole framebuffer, complete from the lower left corner to the upper right

    OPENGL_CALL(glDrawArrays(GL_TRIANGLES, 0, 6));
  }

  // Paired with glfwCreateWindow().
  glfwDestroyWindow(window);

  // Paired with glfwInit().
  glfwTerminate();

  return 0;
}

/*!
 * \brief Create and compile a shader from a source string.
 * \param shader_kind The kind of shader. Could be GL_VERTEX_SHADER or GL_FRAGMENT_SHADER.
 * \param shader_src The source string of the shader.
 * \return The compiled shader ID.
 */
GLuint CreateShader(GLenum shader_kind, const char *shader_src) {
  // Create the shader.
  GLuint shader = glCreateShader(shader_kind);
  glShaderSource(shader, 1, &shader_src, nullptr);
  glCompileShader(shader);

  // Check compile errors.
  GLint err;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &err);

  GLint info_log_len;
  glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &info_log_len);

  if (info_log_len > 0) {
    std::unique_ptr<char[]> err_msg = std::make_unique<char[]>(static_cast<size_t>(info_log_len) + 1);
    glGetShaderInfoLog(shader, info_log_len, nullptr, err_msg.get());
    std::cout << err_msg.get() << std::endl;
    assert(false);
  }

  OPENGL_CHECK_ERROR();

  return shader;
}

/*!
 * \brief Create a program that uses the given compiled vertex and fragment shaders.
 * \param vertex_shader The compiled vertex shader.
 * \param fragment_shader The compiled fragment shader.
 * \return The program ID.
 */
GLuint CreateProgram(GLuint vertex_shader, GLuint fragment_shader) {
  // Create the program and link the shaders.
  GLuint program = glCreateProgram();
  glAttachShader(program, vertex_shader);
  glAttachShader(program, fragment_shader);
  glLinkProgram(program);

  // Check link errors.
  GLint err;
  glGetProgramiv(program, GL_LINK_STATUS, &err);

  GLint info_log_len;
  glGetProgramiv(program, GL_INFO_LOG_LENGTH, &info_log_len);

  if (info_log_len > 0) {
    std::unique_ptr<char[]> err_msg = std::make_unique<char[]>(static_cast<size_t>(info_log_len) + 1);
    glGetProgramInfoLog(program, info_log_len, nullptr, err_msg.get());
    std::cout << err_msg.get() << std::endl;
    assert(false);
  }

  OPENGL_CHECK_ERROR();

  glDetachShader(program, vertex_shader);
  glDetachShader(program, fragment_shader);

  return program;
}

/*!
 * \brief Create a program that uses the given vertex and fragment shader sources.
 * \param vertex_shader_src The vertex shader source.
 * \param fragment_shader_src The fragment shader source.
 * \return The program ID.
 */
GLuint CreateProgram(const char *vertex_shader_src, const char *fragment_shader_src) {
  GLuint vertex_shader = CreateShader(GL_VERTEX_SHADER, vertex_shader_src);
  GLuint fragment_shader = CreateShader(GL_FRAGMENT_SHADER, fragment_shader_src);
  GLuint program = CreateProgram(vertex_shader, fragment_shader);
  glDeleteShader(vertex_shader);
  glDeleteShader(fragment_shader);
  return program;
}
