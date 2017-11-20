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
  while (glGetError() != GL_NO_ERROR);
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
static const char *vertex_shader_text = "#version 330 core\n"
    "in vec2 point; // input to vertex shader\n"
    "void main() {\n"
    "  gl_Position = vec4(point, 0.0, 1.0);\n"
    "}\n";

// This is the main part.
static const char *fragment_shader_text = "#version 330 core\n"
    "uniform int width;\n"
    "uniform int height;\n"
    "uniform sampler2D texture0;\n"
    "uniform sampler2D texture1;\n"
    "out vec4 color;\n"
    "void main() {\n"
    "  // TODO(zhixunt): Calculate pixel index.\n"
    "  ivec2 pixel = ivec2(gl_FragCoord.xy);\n"
    "  color = vec4(\n"
    "    texelFetch(texture0, pixel, 0).r,\n"
    "    texelFetch(texture1, pixel, 0).r,\n"
    "    0.0,\n"
    "    1.0\n"
    "  );\n"
    "}\n";

// The following functions don't need to be understood.
// They are standard practices.
GLuint CreateShader(GLenum shader_kind, const char *shader_src);

GLuint CreateProgram(GLuint vertex_shader, GLuint fragment_shader);

GLuint CreateProgram(const char *vertex_shader, const char *fragment_shader);

class Texture {
 public:
  explicit Texture(const GLfloat *data, GLsizei width, GLsizei height);
  ~Texture();
  Texture(Texture &&other) noexcept;
  Texture(const Texture &other) = delete;
  Texture &operator=(const Texture &other) = delete;

  GLuint texture() const { return texture_; }
  GLsizei width() const { return width_; }
  GLsizei height() const { return height_; }

  void GetData(GLfloat *data) const;

 private:
  static const GLuint kInvalidTexture = static_cast<GLuint>(-1);

  GLuint texture_;
  GLsizei width_;
  GLsizei height_;
};

GLuint NumTextureUnits();

void BindTextureUnit(GLuint unit, GLuint texture);

void BindTextureUnit(GLuint unit, const Texture &texture);

class Workspace {
 public:
  static Workspace &GetInstance();
 private:
  explicit Workspace();
};

int main() {
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
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  GLFWwindow *window = glfwCreateWindow(width, height, "My Title", nullptr,
                                        nullptr);
  if (window == nullptr) {
    std::cout << "glfwCreateWindow() failed!" << std::endl;
    return 1;
  }

  std::cout << "GLFW says OpenGL version is "
            << glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MAJOR)
            << "."
            << glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MINOR)
            << "."
            << glfwGetWindowAttrib(window, GLFW_CONTEXT_REVISION)
            << std::endl;

  // Before using any OpenGL API, we must specify a context.
  glfwMakeContextCurrent(window);

  // Must be called after
  gladLoadGL();
  std::cout << "Opengl says its version is "
            << glGetString(GL_VERSION) << std::endl;

  OPENGL_CHECK_ERROR();

  GLuint program = CreateProgram(vertex_shader_text, fragment_shader_text);

  OPENGL_CALL(glUseProgram(program));

  GLuint vertex_buffer;
  OPENGL_CALL(glGenBuffers(1, &vertex_buffer));
  OPENGL_CALL(glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer));
  OPENGL_CALL(glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices,
                           GL_STATIC_DRAW));

  GLuint vertex_array;
  OPENGL_CALL(glGenVertexArrays(1, &vertex_array));
  OPENGL_CALL(glBindVertexArray(vertex_array));

  auto point_attrib = static_cast<GLuint>(
      glGetAttribLocation(program, "point"));
  OPENGL_CALL(glEnableVertexAttribArray(point_attrib));
  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
  OPENGL_CALL(
      glVertexAttribPointer(point_attrib, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                            nullptr));

  glfwGetFramebufferSize(window, &width, &height);
  auto width_uniform = glGetUniformLocation(program, "width");
  auto height_uniform = glGetUniformLocation(program, "height");
  OPENGL_CALL(glUniform1i(width_uniform, width));
  OPENGL_CALL(glUniform1i(height_uniform, height));
  OPENGL_CALL(glViewport(0, 0, width, height));

  auto texture_size = static_cast<size_t>(width) * height;

  std::vector<GLfloat> texture0_data(texture_size, 0.25f);
  auto texture0 = Texture(texture0_data.data(), width, height);

  std::vector<GLfloat> texture1_data(texture_size, 0.25f);
  auto texture1 = Texture(texture1_data.data(),width, height);

  BindTextureUnit(0, texture0);
  GLint texture0_uniform = glGetUniformLocation(program, "texture0");
  OPENGL_CALL(glUniform1i(texture0_uniform, 0));

  BindTextureUnit(1, texture1);
  GLint texture1_uniform = glGetUniformLocation(program, "texture1");
  OPENGL_CALL(glUniform1i(texture1_uniform, 1));

  while (glfwWindowShouldClose(window) == GL_FALSE) {
    glClear(GL_COLOR_BUFFER_BIT);
    OPENGL_CALL(glDrawArrays(GL_TRIANGLES, 0, 6));
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  {
    // Create frame buffer.
    GLuint frame_buffer = 0;
    OPENGL_CALL(glGenFramebuffers(1, &frame_buffer));
    OPENGL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer));

    auto target_tex = Texture(nullptr, width, height);

    // The depth buffer
    GLuint depthrenderbuffer;
    OPENGL_CALL(glGenRenderbuffers(1, &depthrenderbuffer));
    OPENGL_CALL(glBindRenderbuffer(GL_RENDERBUFFER, depthrenderbuffer));
    OPENGL_CALL(
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width,
                              height));
    OPENGL_CALL(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                          GL_RENDERBUFFER, depthrenderbuffer));

    // Set "renderedTexture" as our colour attachement #0
    OPENGL_CALL(glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                     target_tex.texture(), 0));

    // Set the list of draw buffers.
    GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
    OPENGL_CALL(
        glDrawBuffers(1, DrawBuffers)); // "1" is the size of DrawBuffers

    // Always check that our framebuffer is ok
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      return false;

    // Render to our framebuffer
    // Render on the whole framebuffer, complete from the lower left corner to the upper right
    OPENGL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer));
    OPENGL_CALL(glViewport(0, 0, width, height));

    OPENGL_CALL(glClear(GL_COLOR_BUFFER_BIT));
    OPENGL_CALL(glDrawArrays(GL_TRIANGLES, 0, 6));

    std::vector<GLfloat> retrieved_data(static_cast<size_t>(width * height));
    target_tex.GetData(retrieved_data.data());

    std::cout << "Wat" << std::endl;
  }

  // Paired with glfwCreateWindow().
  glfwDestroyWindow(window);

  // Paired with glfwInit().
  glfwTerminate();

  return 0;
}

/*!
 * \brief Create and compile a shader from a source string.
 * \param shader_kind The kind of shader.
 * Could be GL_VERTEX_SHADER or GL_FRAGMENT_SHADER.
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
    std::unique_ptr<char[]> err_msg(new char[info_log_len + 1]);
    glGetShaderInfoLog(shader, info_log_len, nullptr, err_msg.get());
    std::cout << err_msg.get() << std::endl;
    assert(false);
  }

  OPENGL_CHECK_ERROR();

  return shader;
}

/*!
 * \brief Create a program that uses the given vertex and fragment shaders.
 * \param vertex_shader The **compiled** vertex shader.
 * \param fragment_shader The **compiled** fragment shader.
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
    std::unique_ptr<char[]> err_msg(new char[info_log_len + 1]);
    glGetProgramInfoLog(program, info_log_len, nullptr, err_msg.get());
    std::cout << err_msg.get() << std::endl;
    assert(false);
  }

  OPENGL_CHECK_ERROR();

  OPENGL_CALL(glDetachShader(program, vertex_shader));
  OPENGL_CALL(glDetachShader(program, fragment_shader));

  return program;
}

/*!
 * \brief Create a program that uses the given vertex and fragment shader.
 * \param vertex_shader The vertex shader **source**.
 * \param fragment_shader The fragment shader **source**.
 * \return The program ID.
 */
GLuint CreateProgram(const char *vertex_shader, const char *fragment_shader) {
  // Create and compile the shaders.
  GLuint vertex_shader_id = CreateShader(GL_VERTEX_SHADER, vertex_shader);
  GLuint fragment_shader_id = CreateShader(GL_FRAGMENT_SHADER, fragment_shader);

  // Link the shaders and create the program.
  GLuint program = CreateProgram(vertex_shader_id, fragment_shader_id);

  OPENGL_CALL(glDeleteShader(vertex_shader_id));
  OPENGL_CALL(glDeleteShader(fragment_shader_id));

  return program;
}

Texture::Texture(const GLfloat *data, GLsizei width, GLsizei height)
    : texture_(kInvalidTexture), width_(width), height_(height) {

  // Create a texture.
  OPENGL_CALL(glGenTextures(1, &texture_));

  // Bind to temporary unit.
  BindTextureUnit(NumTextureUnits() - 1, texture_);

  // Similar to cudaMemcpy.
  OPENGL_CALL(glTexImage2D(GL_TEXTURE_2D, /*level=*/0, GL_RED,
                           width_, height_, /*border=*/0,
                           GL_RED, GL_FLOAT, data));

  // TODO(zhixunt): What are these?
  OPENGL_CALL(
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
  OPENGL_CALL(
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
  OPENGL_CALL(
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
  OPENGL_CALL(
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
}

Texture::Texture(Texture &&other) noexcept
    : texture_(other.texture_), width_(other.width_), height_(other.height_) {
  other.texture_ = kInvalidTexture;
}

Texture::~Texture() {
  if (texture_ != kInvalidTexture) {
    glDeleteTextures(1, &texture_);
    texture_ = kInvalidTexture;
  }
}

void Texture::GetData(GLfloat *data) const {
  BindTextureUnit(NumTextureUnits() - 1, *this);
  glGetTexImage(GL_TEXTURE_2D, /*level=*/0, GL_RED, GL_FLOAT, data);
}

GLuint NumTextureUnits() {
  GLint num_units;
  OPENGL_CALL(glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &num_units));
  return static_cast<GLuint>(num_units);
}

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
void BindTextureUnit(GLuint unit, GLuint texture) {
  OPENGL_CALL(glActiveTexture(GL_TEXTURE0 + unit));
  OPENGL_CALL(glBindTexture(GL_TEXTURE_2D, texture));
}

void BindTextureUnit(GLuint unit, const Texture &texture) {
  BindTextureUnit(unit, texture.texture());
}

Workspace &Workspace::GetInstance() {
  static std::unique_ptr<Workspace> instance_(new Workspace);
  return *instance_;
}

Workspace::Workspace() {

}
