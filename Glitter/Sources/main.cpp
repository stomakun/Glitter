#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <vector>
#include <random>

int opencl(GLfloat *a, GLfloat *b, GLfloat *c, int iters, unsigned int N, int gpu = 1);

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

void OPENGL_ABSORB_ERRORS() {
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

// This is the main part.
static const char *fragment_shader_text = "#version 330 core\n"
    "uniform sampler2D A;\n"
    "uniform sampler2D B;\n"
    "uniform int N;\n"
    "out float color;\n"
    "void main() {\n"
    "  ivec2 pixel = ivec2(gl_FragCoord.xy);\n"
    "  int idx = pixel.x;\n"
    "  int row = idx / N;\n"
    "  int col = idx % N;\n"
    "  color = 0.0;\n"
    "  for (int i = 0; i < N; i++) {\n"
    "    float a = texelFetch(A, ivec2(row * N + i, 0), 0).r;\n"
    "    float b = texelFetch(B, ivec2(i * N + col, 0), 0).r;\n"
    "    color += a * b;\n"
    "  }\n"
    "}\n";

/*!
 * \brief An OpenGL program, composed of a vertex shader and a fragment shader.
 * In TVM, every program has the same vertex shader.
 * So a program just corresponds to a fragment shader.
 * A program can only be created by the workspace.
 * This class is just a wrapper over an OpenGL program ID.
 */
class Program {
 public:
  // Move constructor.
  Program(Program &&other) noexcept;

  // Cannot be copied.
  Program(const Program &other) = delete;

  // Cannot be assigned.
  Program &operator=(const Program &other) = delete;

  // Destructor.
  ~Program();

 private:
  friend class Workspace;

  // Only a workspace can construct a program.
  explicit Program(GLuint program);

  // The internal OpenGL program ID.
  GLuint program_;

  static const GLuint kInvalidProgram = static_cast<GLuint>(-1);
};

/*!
 * An OpenGL texture represents a chunk of GPU memory.
 * This is the way we represent tensors.
 * We always use 2D textures.
 */
class Texture {
 public:
  ~Texture();

  Texture(Texture &&other) noexcept;

  Texture(const Texture &other) = delete;

  Texture &operator=(const Texture &other) = delete;

  GLsizei width() const { return width_; }

  GLsizei height() const { return height_; }

  void GetData(GLfloat *data) const;

 private:
  friend class Workspace;

  explicit Texture(const GLfloat *data, GLsizei width, GLsizei height);

  GLuint texture() const { return texture_; }

  static const GLuint kInvalidTexture = static_cast<GLuint>(-1);

  GLuint texture_;
  GLsizei width_;
  GLsizei height_;
};

/*!
 * The OpenGL workspace.
 * This is a global singleton.
 */
class Workspace {
 public:
  // Get singleton instance.
  static Workspace &GetInstance();

  // Cannot be moved.
  Workspace(Workspace &&other) = delete;

  // Cannot be copied.
  Workspace(const Workspace &other) = delete;

  // Cannot be assigned.
  Workspace &operator=(const Workspace &other) = delete;

  ~Workspace();

  // Compile a fragment shader and create a program.
  Program CreateProgram(const char *fragment_shader_src);

  // Create a texture with the given data.
  Texture CreateTexture(const GLfloat *data, GLsizei width, GLsizei height);

  // Render to a texture.
  void Render(const Program &program,
              const std::vector<std::pair<std::string, Texture *>> &inputs,
              const std::vector<std::pair<std::string, int>> &uniforms,
              Texture *output,
              int niters);

  // Render to the main window.
  // This is for debugging purposes.
  void Render(const Program &program,
              const std::vector<std::pair<std::string, Texture *>> &inputs);

  static const int kWindowWidth = 640;

  static const int kWindowHeight = 480;

 private:
  friend class Texture;

  explicit Workspace();

  GLuint NumTextureUnits();

  void BindTextureUnit(GLuint unit, GLuint texture);

  void BindTextureUnit(GLuint unit, const Texture &texture);

  GLuint CreateShader(GLenum shader_kind, const char *shader_src);

  Program CreateProgram(GLuint fragment_shader);

  // Don't need to change this.
  // We want to draw 2 giant triangles that cover the whole screen.
  struct Vertex {
    float x, y;
  };

  static constexpr size_t kNumVertices = 6;

  static const Vertex vertices[kNumVertices];

  static const char *vertex_shader_text_;

 public:
  GLFWwindow *window_;
  GLuint vertex_shader_;
};

void TestRenderToWindow() {
  Workspace &workspace = Workspace::GetInstance();

  GLint width, height;
  glfwGetFramebufferSize(workspace.window_, &width, &height);

  auto texture_size = static_cast<size_t>(width) * height;

  std::vector<GLfloat> texture0_data(texture_size, 0.25f);
  auto texture0 = workspace.CreateTexture(texture0_data.data(), width, height);

  std::vector<GLfloat> texture1_data(texture_size, 0.25f);
  auto texture1 = workspace.CreateTexture(texture1_data.data(), width, height);

  Program program = workspace.CreateProgram(fragment_shader_text);

  while (glfwWindowShouldClose(workspace.window_) == GL_FALSE) {
    workspace.Render(
        program, {
            {"texture0", &texture0},
            {"texture1", &texture1}
        }
    );
    glfwSwapBuffers(workspace.window_);
    glfwPollEvents();
  }
}

void TestRenderToTexture(int N, int niters) {
  Workspace &workspace = Workspace::GetInstance();

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(1.0f, 2.0f);

  GLint width = N * N;
  GLint height = 1;
  auto texture_size = static_cast<size_t>(width) * height;

  std::vector<GLfloat> texture0_data(texture_size, 0.0f);
  for (size_t i = 0; i != texture_size; ++i) {
    texture0_data[i] = dist(mt);
  }
  auto texture0 = workspace.CreateTexture(texture0_data.data(), width, height);

  std::vector<GLfloat> texture1_data(texture_size, 0.0f);
  for (size_t i = 0; i != texture_size; ++i) {
    texture1_data[i] = dist(mt);
  }
  auto texture1 = workspace.CreateTexture(texture1_data.data(), width, height);

  Program program = workspace.CreateProgram(fragment_shader_text);

  auto target_texture = workspace.CreateTexture(nullptr, width, height);

  workspace.Render(
      program, {
          {"A", &texture0},
          {"B", &texture1}
      }, {
          {"N", N}
      },
      &target_texture,
      niters
  );

  std::vector<GLfloat> retrieved_data(static_cast<size_t>(width * height));
  target_texture.GetData(retrieved_data.data());

  std::vector<GLfloat> cpu_result(static_cast<size_t>(width * height));
  auto cpu_start = std::chrono::system_clock::now();
  for (int iter = 0; iter < niters; ++iter) {
    for (int row = 0; row != N; ++row) {
      for (int col = 0; col != N; ++col) {
        cpu_result[row * N + col] = 0.0f;
        for (int i = 0; i != N; ++i) {
          GLfloat a = texture0_data[row * N + i];
          GLfloat b = texture1_data[i * N + col];
          cpu_result[row * N + col] += a * b;
        }
      }
    }
  }
  auto cpu_end = std::chrono::system_clock::now();

  for (size_t i = 0; i < retrieved_data.size(); ++i) {
    assert(std::abs(retrieved_data[i] - cpu_result[i]) < 0.001f);
  }

  std::cout << "cpu:    "
            << (std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count() / niters)
            << std::endl;

  std::vector<GLfloat> cl_result(static_cast<size_t>(width * height));
    opencl(texture0_data.data(), texture1_data.data(), cl_result.data(), niters, (unsigned int)N);
    opencl(texture0_data.data(), texture1_data.data(), cl_result.data(), niters, (unsigned int)N, 0);

  for (size_t i = 0; i < retrieved_data.size(); ++i) {
    assert(std::abs(cl_result[i] - cpu_result[i]) < 0.001f);
  }
}

int main() {
  Workspace::GetInstance();

  TestRenderToTexture(64, /*niters=*/100);

  return 0;
}

Program::Program(Program &&other) noexcept : program_(other.program_) {
  other.program_ = kInvalidProgram;
}

Program::Program(GLuint program) : program_(program) {}

Program::~Program() {
  if (program_ != kInvalidProgram) {
    glDeleteProgram(program_);
    program_ = kInvalidProgram;
  }
}

Texture::Texture(const GLfloat *data, GLsizei width, GLsizei height)
    : texture_(kInvalidTexture), width_(width), height_(height) {
  auto &workspace = Workspace::GetInstance();

  // Create a texture.
  OPENGL_CALL(glGenTextures(1, &texture_));

  std::clog << "Created texture [" << texture_ << "]" << std::endl;

  // Bind to temporary unit.
  workspace.BindTextureUnit(workspace.NumTextureUnits() - 1, texture_);

  // Similar to cudaMemcpy.
  OPENGL_CALL(glTexImage2D(GL_TEXTURE_2D, /*level=*/0, GL_RGBA32F,
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
    std::clog << "Deleting texture [" << texture_ << "]" << std::endl;
    OPENGL_CALL(glDeleteTextures(1, &texture_));
    texture_ = kInvalidTexture;
  }
}

void Texture::GetData(GLfloat *data) const {
  auto &workspace = Workspace::GetInstance();
  workspace.BindTextureUnit(workspace.NumTextureUnits() - 1, texture_);

  glGetTexImage(GL_TEXTURE_2D, /*level=*/0, GL_RED, GL_FLOAT, data);
}

GLuint Workspace::NumTextureUnits() {
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
void Workspace::BindTextureUnit(GLuint unit, GLuint texture) {
  OPENGL_CALL(glActiveTexture(GL_TEXTURE0 + unit));
  OPENGL_CALL(glBindTexture(GL_TEXTURE_2D, texture));
}

void Workspace::BindTextureUnit(GLuint unit, const Texture &texture) {
  BindTextureUnit(unit, texture.texture());
}

Workspace &Workspace::GetInstance() {
  static std::unique_ptr<Workspace> instance_(new Workspace);
  return *instance_;
}

Workspace::Workspace() {
  // Set an error handler.
  // This can be called before glfwInit().
  glfwSetErrorCallback(&GlfwErrorCallback);

  // Initialize GLFW.
  if (glfwInit() != GL_TRUE) {
    std::cout << "glfwInit() failed!" << std::endl;
    assert(false);
  }

  // Create a window.
  // TODO(zhixunt): GLFW allows us to create an invisible window.
  // TODO(zhixunt): On retina display, window size is different from framebuffer size.
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  window_ = glfwCreateWindow(kWindowWidth, kWindowHeight, "", nullptr, nullptr);
  if (window_ == nullptr) {
    std::cout << "glfwCreateWindow() failed!" << std::endl;
    assert(false);
  }

  std::cout << "GLFW says OpenGL version: "
            << glfwGetWindowAttrib(window_, GLFW_CONTEXT_VERSION_MAJOR)
            << "."
            << glfwGetWindowAttrib(window_, GLFW_CONTEXT_VERSION_MINOR)
            << "."
            << glfwGetWindowAttrib(window_, GLFW_CONTEXT_REVISION)
            << std::endl;

  // Before using any OpenGL API, we must specify a context.
  glfwMakeContextCurrent(window_);

  // Must be called after creating GLFW window.
  gladLoadGL();

  std::cout << "Opengl says version: " << glGetString(GL_VERSION) << std::endl;

  OPENGL_CHECK_ERROR();

  // We always render the same vertices and triangles.
  GLuint vertex_buffer;
  OPENGL_CALL(glGenBuffers(1, &vertex_buffer));
  OPENGL_CALL(glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer));
  OPENGL_CALL(glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices,
                           GL_STATIC_DRAW));

  GLuint vertex_array;
  OPENGL_CALL(glGenVertexArrays(1, &vertex_array));
  OPENGL_CALL(glBindVertexArray(vertex_array));
  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);

  // We always use the same vertex shader.
  vertex_shader_ = CreateShader(GL_VERTEX_SHADER, vertex_shader_text_);
}

Workspace::~Workspace() {
  // Paired with glfwCreateWindow().
  glfwDestroyWindow(window_);

  // Paired with glfwInit().
  glfwTerminate();
}

/*!
 * \brief Create a program that uses the given vertex and fragment shader.
 * \param fragment_shader The fragment shader **source**.
 * \return The program ID.
 */
Program Workspace::CreateProgram(const char *fragment_shader_src) {
  // Create and compile the shaders.
  GLuint fragment_shader = CreateShader(GL_FRAGMENT_SHADER,
                                        fragment_shader_src);

  // Link the shaders and create the program.
  Program program = CreateProgram(fragment_shader);

  OPENGL_CALL(glDeleteShader(fragment_shader));

  return program;
}

void Workspace::Render(
    const Program &program,
    const std::vector<std::pair<std::string, Texture *>> &inputs,
    const std::vector<std::pair<std::string, int>> &uniforms,
    Texture *output,
    int niters) {
  if (inputs.size() + 2 > NumTextureUnits()) {
    std::cerr << "Too many inputs!" << std::endl;
    assert(false);
  }

  OPENGL_CALL(glUseProgram(program.program_));

  // Create frame buffer.
  GLuint frame_buffer;
  OPENGL_CALL(glGenFramebuffers(1, &frame_buffer));
  OPENGL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer));

  // Set "renderedTexture" as our colour attachement #0
  OPENGL_CALL(glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   output->texture(), 0));

  // Set the list of draw buffers.
  GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
  // "1" is the size of DrawBuffers.
  OPENGL_CALL(glDrawBuffers(1, DrawBuffers));

  // Always check that our framebuffer is ok
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    std::cout << "Framebuffer not complete." << std::endl;
    assert(false);
  }

  // Tell the fragment shader what input textures to use.
  for (GLuint unit = 0; unit != inputs.size(); ++unit) {
    const std::string &name = inputs[unit].first;
    Texture *texture = inputs[unit].second;

    BindTextureUnit(unit, *texture);

    GLint texture_uniform = glGetUniformLocation(program.program_,
                                                 name.c_str());
    OPENGL_CALL(glUniform1i(texture_uniform, unit));
  }

  // Tell the fragment shader about uniforms.
  for (auto &uniform : uniforms) {
    const std::string &name = uniform.first;
    int value = uniform.second;
    GLint shader_uniform = glGetUniformLocation(program.program_, name.c_str());
    OPENGL_CALL(glUniform1i(shader_uniform, value));
  }

  OPENGL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer));
  OPENGL_CALL(glViewport(0, 0, output->width(), output->height()));

  auto opengl_start = std::chrono::system_clock::now();
  for (int iter = 0; iter < niters; ++iter) {
    OPENGL_CALL(glClear(GL_COLOR_BUFFER_BIT));
    OPENGL_CALL(glDrawArrays(GL_TRIANGLES, 0, 6));
    glFinish();
  }

  glDeleteFramebuffers(1, &frame_buffer);

  auto opengl_end = std::chrono::system_clock::now();
  std::cout << "opengl: "
            << (std::chrono::duration_cast<std::chrono::microseconds>(opengl_end - opengl_start).count() / niters)
            << std::endl;
}

void Workspace::Render(
    const Program &program,
    const std::vector<std::pair<std::string, Texture *>> &inputs) {
  if (inputs.size() + 1 > NumTextureUnits()) {
    std::cerr << "Too many inputs!" << std::endl;
    assert(false);
  }

  OPENGL_CALL(glUseProgram(program.program_));

  // Tell the fragment shader what input textures to use.
  for (GLuint unit = 0; unit != inputs.size(); ++unit) {
    const std::string &name = inputs[unit].first;
    Texture *texture = inputs[unit].second;

    BindTextureUnit(unit, *texture);

    GLint texture_uniform = glGetUniformLocation(program.program_,
                                                 name.c_str());
    OPENGL_CALL(glUniform1i(texture_uniform, unit));
  }

  // Framebuffer 0 means the window.
  OPENGL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, 0));

  // FIll the entire window.
  GLint width, height;
  glfwGetFramebufferSize(window_, &width, &height);
  OPENGL_CALL(glViewport(0, 0, width, height));

  OPENGL_CALL(glClear(GL_COLOR_BUFFER_BIT));
  OPENGL_CALL(glDrawArrays(GL_TRIANGLES, 0, 6));
}

/*!
 * \brief Create and compile a shader from a source string.
 * \param shader_kind The kind of shader.
 * Could be GL_VERTEX_SHADER or GL_FRAGMENT_SHADER.
 * \param shader_src The source string of the shader.
 * \return The compiled shader ID.
 */
GLuint Workspace::CreateShader(GLenum shader_kind, const char *shader_src) {
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
 * \param fragment_shader The **compiled** fragment shader.
 * \return The program ID.
 */
Program Workspace::CreateProgram(GLuint fragment_shader) {
  // Create the program and link the shaders.
  GLuint program = glCreateProgram();
  glAttachShader(program, vertex_shader_);
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

  OPENGL_CALL(glDetachShader(program, vertex_shader_));
  OPENGL_CALL(glDetachShader(program, fragment_shader));

  auto point_attrib = GLuint(glGetAttribLocation(program, "point"));
  OPENGL_CALL(glEnableVertexAttribArray(point_attrib));

  OPENGL_CALL(glVertexAttribPointer(point_attrib, 2, GL_FLOAT, GL_FALSE,
                                    sizeof(Vertex), nullptr));

  return Program(program);
}

Texture Workspace::CreateTexture(const GLfloat *data, GLsizei width,
                                 GLsizei height) {
  return Texture(data, width, height);
}

// Don't need to change this.
// The vertex shader only needs to take in the triangle points.
// No need for point transformations.
const char *Workspace::vertex_shader_text_ = "#version 330 core\n"
    "in vec2 point; // input to vertex shader\n"
    "void main() {\n"
    "  gl_Position = vec4(point, 0.0, 1.0);\n"
    "}\n";

const Workspace::Vertex Workspace::vertices[kNumVertices] = {
    {-1.f, -1.f},
    {1.0f, -1.f},
    {1.0f, 1.0f},
    {-1.f, -1.f},
    {-1.f, 1.0f},
    {1.0f, 1.0f},
};