// Unicode braille waterfall

#include <boost/lockfree/spsc_queue.hpp>
#include <fftw3.h>
#include <jack/jack.h>
#include <sndfile.hh>

#include <algorithm>
#include <cassert>
#include <charconv>
#include <chrono>
#include <concepts>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numbers>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace {

using brl_t = uint8_t;

constexpr brl_t brl(unsigned dots)
{
  brl_t mask = 0;
  while (dots != 0) {
    const unsigned d = dots % 10;
    if (d >= 1 && d <= 8) mask |= brl_t{1} << (d - 1);
    dots /= 10;
  }
  return mask;
}

std::string to_utf8(std::span<brl_t> cells)
{
  std::string s;
  s.reserve(cells.size() * 3);

  for (brl_t cell: cells) {
    auto cp = static_cast<char32_t>(0x2800u + cell);
    s.push_back(0xE0 | ((cp >> 12) & 0x0F));
    s.push_back(0x80 | ((cp >> 6) & 0x3F));
    s.push_back(0x80 | (cp & 0x3F));
  }

  return s;
}

float hann(size_t n, size_t N)
{
  constexpr auto tau = 2.0f * std::numbers::pi_v<float>;
  return 0.5f - 0.5f * std::cos(tau * n / (N - 1));
}

class FFT
{
  std::vector<float> in;
  std::vector<float> win;
  float win_sum;
  std::vector<fftwf_complex> out;
  fftwf_plan plan;

public:
  FFT(size_t n, float (*window)(size_t, size_t))
  : in(n)
  , win{ std::views::iota(size_t{0}, n)
       | std::views::transform([=](size_t i) { return window(i, n); })
       | std::ranges::to<std::vector>()
       }
  , win_sum{std::ranges::fold_left(win, 0.0f, std::plus{})}
  , out(n / 2 + 1)
  , plan{fftwf_plan_dft_r2c_1d(n, in.data(), out.data(), FFTW_ESTIMATE)}
  {}
  FFT(FFT const &) = delete;
  FFT &operator=(FFT const &) = delete;
  ~FFT() { fftwf_destroy_plan(plan); }

  std::span<float> input() { return in; }

  void execute()
  {
    std::ranges::transform(in, win, in.begin(), std::multiplies{});
    fftwf_execute(plan);
  }

  std::span<const fftwf_complex> output() const { return out; }

  auto dbfs() const
  {
    auto to_dbfs = [win_sum = win_sum, max_k = out.size() - 1](auto&& t)
    {
      auto [k, value] = t;
      float mag = std::hypot(value[0], value[1]);
      if (k != 0 && k != max_k) mag *= 2.0f;
      const float amplitude = mag / win_sum;
      return 20.0f * std::log10(std::max(amplitude, std::numeric_limits<float>::epsilon()));
    };
    return out | std::views::enumerate | std::views::transform(to_dbfs);
  }
};

inline auto mono(size_t channels)
{
  constexpr auto mean = [](auto frame)
  {
    using V = std::ranges::range_value_t<decltype(frame)>;
    return std::ranges::fold_left(frame, V{}, std::plus<>{}) / frame.size();
  };
  return std::views::chunk(channels) | std::views::transform(mean);
}

struct AudioSource
{
  virtual ~AudioSource() = default;
  virtual size_t channels() const = 0;
  virtual unsigned int samplerate() const = 0;
  virtual size_t readf(float *interleaved, size_t frames) = 0;
};

class SoundFile final : public AudioSource
{
  SndfileHandle sf;

public:
  SoundFile(std::string path)
  : sf(path)
  { if (sf.error()) throw std::runtime_error(sf.strError()); }

  size_t channels() const override { return sf.channels(); }
  unsigned int samplerate() const override { return sf.samplerate(); }
  size_t readf(float *interleaved, size_t frames) override
  { return sf.readf(interleaved, frames); }
};

class JACK final : public AudioSource
{
  jack_client_t* client{};
  std::vector<jack_port_t*> inports;   // non-RT
  std::vector<float> tmp;              // non-RT, reused in RT
  unsigned int fs{};

  boost::lockfree::spsc_queue<float, boost::lockfree::capacity<1 << 18>> q;

  static int process(jack_nframes_t nframes, void* arg)
  {
    auto& self = *static_cast<JACK*>(arg);
    const size_t C = self.inports.size();
    const size_t n = size_t(nframes) * C;
    assert(n <= self.tmp.size());

    for (auto [ch, port]: self.inports | std::views::enumerate) {
      auto* in = static_cast<const float*>(jack_port_get_buffer(port, nframes));
      for (jack_nframes_t i = 0; i < nframes; ++i)
        self.tmp[C * i + ch] = in[i];
    }

    self.q.push(self.tmp.begin(), self.tmp.end());
    return 0;
  }

  static int bufsize(jack_nframes_t nframes, void* arg)
  {
    auto& self = *static_cast<JACK*>(arg);
    auto new_size = self.inports.size() * nframes;
    if (self.tmp.size() < new_size) self.tmp.resize(new_size);
    return 0;
  }

public:
  explicit JACK(size_t inputs = 1)
  : client{jack_client_open("braille-scope", JackNoStartServer, nullptr)}
  , inports(inputs)
  {
    if (!client) throw std::runtime_error("Failed to open JACK client");
    fs = jack_get_sample_rate(client);

    for (size_t ch = 0; ch < inputs; ++ch) {
      auto name = "in_" + std::to_string(ch + 1);
      inports[ch] = jack_port_register(client, name.c_str(),
                                       JACK_DEFAULT_AUDIO_TYPE, JackPortIsInput, 0);
    }

    tmp.resize(inputs * jack_get_buffer_size(client));

    jack_set_process_callback(client, &JACK::process, this);
    jack_set_buffer_size_callback(client, &JACK::bufsize, this);
    jack_activate(client);
  }

  ~JACK() override
  {
    jack_deactivate(client);
    jack_client_close(client);
  }

  size_t channels() const override { return inports.size(); }
  unsigned int samplerate() const override { return fs; }

  size_t readf(float* interleaved, size_t frames) override
  {
    const size_t want_samples = frames * channels();

    while (q.read_available() < want_samples) {
      const size_t avail_frames = q.read_available() / channels();
      const auto missing_frames = frames - avail_frames;

      using clock = std::chrono::steady_clock;
      auto dt = clock::duration(std::chrono::seconds(missing_frames)) / samplerate();

      dt -= std::chrono::microseconds(200);
      if (dt > std::chrono::microseconds(0)) std::this_thread::sleep_for(dt);
      else std::this_thread::yield();
    }

    q.pop(interleaved, want_samples);
    return frames;
  }
};

template <class R>
concept resample_input =
  std::ranges::random_access_range<R> &&
  std::ranges::sized_range<R> &&
  std::convertible_to<std::ranges::range_reference_t<R>, float>;

std::vector<size_t> make_edges(
  size_t size, size_t bins, float bin_scale, auto&& i_to_f
)
{
  auto f_to_k = [=](float f) -> size_t {
    long k = std::lround(f * bin_scale);
    return size_t(std::clamp<long>(k, 0, long(bins - 1)));
  };

  auto edges = std::views::iota(size_t{0}, size + 1)
             | std::views::transform(std::forward<decltype(i_to_f)>(i_to_f))
             | std::views::transform(f_to_k)
             | std::ranges::to<std::vector>();

  for (size_t i = 1; i < edges.size(); ++i)
    edges[i] = std::max(edges[i], edges[i - 1]);

  return edges;
}

inline std::vector<size_t> linear(
  size_t size, size_t bins, float bin_scale, float f_min, float f_max
)
{
  const auto a = (f_max - f_min) / size;
  return make_edges(size, bins, bin_scale, [f_min, a](size_t i) {
    return f_min + a * i;
  });
}

inline std::vector<size_t> logarithmic(
  size_t size, size_t bins, float bin_scale, float f_min, float f_max
)
{
  const auto a = std::log(f_max / f_min) / size;
  return make_edges(size, bins, bin_scale, [f_min, a](size_t i) {
    return f_min * std::exp(a * i);
  });
}

struct log_step {
  float octaves_per_col = 1.0f / 12.0f; // default: 12 bins/octave

  std::vector<size_t> operator()(
    size_t size, size_t fft_bins, float bin_scale, float f_min, float f_max
  ) const
  {
    return make_edges(size, fft_bins, bin_scale, [=, this](size_t i)
      { return std::min(f_min * std::exp2(octaves_per_col * i), f_max); }
    );
  }
};

auto resample(resample_input auto&& in, std::span<const size_t> edges, auto&& out)
{
  assert(edges.size() >= 2);
  auto b = in.begin();
  for (auto [k0, k1] : edges | std::views::pairwise)
    *out++ = *std::max_element(b + k0, b + k1 + 1); // inclusive

  return out;
}

std::string braille_glyphs(std::span<const float> dbfs, float min_dbfs, float max_dbfs)
{
  auto level4 = [&](float d) -> int {
    auto x01 = std::clamp((d - min_dbfs) / (max_dbfs - min_dbfs), 0.0f, 1.0f);
    return std::min(4, int(std::floor(x01 * 5.0f))); // 0..4
  };

  std::vector<brl_t> cells;
  cells.reserve((dbfs.size() + 1) / 2);

  for (auto lr: dbfs | std::views::chunk(2)) {
    const int L = level4(lr[0]);
    const int R = lr.size() == 2 ? level4(lr[1]) : 0;
    brl_t mask = 0;

    if(L >= 1) mask |= brl(1);
    if(L >= 2) mask |= brl(2);
    if(L >= 3) mask |= brl(3);
    if(L == 4) mask |= brl(7);
    if(R >= 1) mask |= brl(4);
    if(R >= 2) mask |= brl(5);
    if(R >= 3) mask |= brl(6);
    if(R == 4) mask |= brl(8);

    cells.push_back(mask);
  }

  return to_utf8(cells);
}

int waterfall(
  AudioSource &src, auto&& init_edges,
  unsigned int fps = 30,
  unsigned int width = 79,
  float display_min_dbfs = -60.0f, float display_max_dbfs = 0.0f,
  float min_freq = 20.0f, float max_freq = -1.0f,
  bool stereo = false
)
{
  const size_t hop  = src.samplerate() / fps;
  const size_t N    = std::bit_ceil(std::bit_ceil(hop + 1) + 1);

  const float Fs  = float(src.samplerate());
  const float nyq = 0.5f * Fs;
  if (max_freq <= 0.0f || max_freq > nyq) max_freq = nyq;
  min_freq = std::max(1.0f, std::min(min_freq, max_freq * 0.999f));
  FFT fft(N, hann);

  std::vector<float> line(2 * width, std::numeric_limits<float>::lowest());

  const float bin_scale = float(N) / Fs;

  using clock = std::chrono::steady_clock;
  auto time = clock::now();
  const auto hop_duration = clock::duration(std::chrono::seconds(hop)) / src.samplerate();
  const size_t overlap = N - hop;

  std::vector<float> interleaved(N * src.channels());
  if (src.readf(interleaved.data(), N) < N) return EXIT_SUCCESS;

  auto edges = init_edges(stereo ? width / 2 * 2 : line.size(), fft.output().size(), bin_scale, min_freq, max_freq);
  do {
    std::putchar('\n');

    if (stereo) {
      const int half_glyphs = int(width / 2);
      const int mid_glyph   = int(width % 2);          // 1 => insert one blank glyph
      const int cols        = 2 * half_glyphs;
      const int mid_cols    = 2 * mid_glyph;           // 0 or 2 columns (one glyph)

      // Left FFT
      for (size_t i = 0; i < N; ++i) fft.input()[i] = interleaved[2 * i + 0];
      fft.execute();
      resample(fft.dbfs(), edges, line.begin());
      std::reverse(line.begin(), line.begin() + cols);

      // Optional center blank glyph (2 columns)
      if (mid_cols) {
        line[cols + 0] = std::numeric_limits<float>::lowest();
        line[cols + 1] = std::numeric_limits<float>::lowest();
      }

      // Right FFT
      for (size_t i = 0; i < N; ++i) fft.input()[i] = interleaved[2 * i + 1];
      fft.execute();
      resample(fft.dbfs(), edges, line.begin() + cols + mid_cols);
    } else {
      std::ranges::copy(interleaved | mono(src.channels()), fft.input().begin());
      fft.execute();
      resample(fft.dbfs(), edges, line.begin());
    }

    std::fputs(braille_glyphs(line, display_min_dbfs, display_max_dbfs).c_str(), stdout);
    fflush(stdout);

    time += hop_duration;
    std::this_thread::sleep_for(time - clock::now());

    std::memmove(interleaved.data(), interleaved.data() + hop * src.channels(),
      overlap * src.channels() * sizeof(float)
    );
  } while (src.readf(interleaved.data() + overlap * src.channels(), hop) == hop);

  return EXIT_SUCCESS;
}

constexpr std::optional<float> parse_freq(std::string_view sv, float a4 = 440.0f)
{
  if (sv.empty()) return std::nullopt;
  
  if (char c = sv.front(); (c >= 'A' && c <= 'G') || (c >= 'a' && c <= 'g')) {
    static constexpr int semis[] = { 9, 11, 0, 2, 4, 5, 7 };
    auto semi = semis[c >= 'a' ? c - 'a' : c - 'A'];

    int acc = 0;
    auto t = sv.substr(1);
    while (!t.empty()) {
      unsigned char c = (unsigned char)t.front();

      if (c == 'b' || c == 'B') { acc -= 1; t.remove_prefix(1); continue; }
      if (c == '#')             { acc += 1; t.remove_prefix(1); continue; }
      if (c == 's' || c == 'S') { acc += 1; t.remove_prefix(1); continue; }

      // UTF-8 ♭ (U+266D): E2 99 AD
      // UTF-8 ♯ (U+266F): E2 99 AF
      if (t.size() >= 3 &&
          (unsigned char)t[0] == 0xE2 && (unsigned char)t[1] == 0x99 &&
          (unsigned char)t[2] == 0xAD) {
        acc -= 1; t.remove_prefix(3); continue;
      }
      if (t.size() >= 3 &&
          (unsigned char)t[0] == 0xE2 && (unsigned char)t[1] == 0x99 &&
          (unsigned char)t[2] == 0xAF) {
        acc += 1; t.remove_prefix(3); continue;
      }

      break;
    }
    int octave{};
    auto [p, ec] = std::from_chars(t.data(), t.data() + t.size(), octave);
    if (ec != std::errc{}) return std::nullopt;
    int midi = (octave + 1) * 12 + semi + acc;
    return a4 * std::exp2((float(midi) - 69.0f) / 12.0f);
  }
  if consteval {
    unsigned freq{};
    auto [p, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), freq);
    if (ec != std::errc{}) return std::nullopt;
    if (p != sv.data() + sv.size()) return std::nullopt;
    return freq;
  }
  float freq{};
  auto [p, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), freq);
  if (ec != std::errc{}) return std::nullopt;
  return freq;
}

static_assert(parse_freq("a4") == 440.0f);
static_assert(parse_freq("440") == 440.0f);

} // namespace

int main(int argc, char* argv[])
{
  auto usage = [=]() -> int {
    std::fprintf(
      stdout,
      "usage:\n"
      "  %s [options] (--jack | <filename>)\n\n"
      "options:\n"
      "  --fps N                 Frames per second (default: 30)\n"
      "  --min-freq HZ           Min displayed frequency (default: 20)\n"
      "  --max-freq HZ           Max displayed frequency (default: nyquist)\n"
      "  --min-dbfs DB           Min displayed level in dBFS (default: -60)\n"
      "  --max-dbfs DB           Max displayed level in dBFS (default: 0)\n"
      "  --width N               Output width in braille glyphs (default: 79)\n"
      "  --linear                Use a linear mapping\n"
      "  --cents-per-bin CENTS   Log mapping: CENTS per column\n"
      "  --bins-per-octave N     Log mapping: N columns per octave\n"
      "  --jack                  Use JACK input instead of a file\n"
      "  --stereo                Mirror 2 channels around the center axis\n"
      "  --help                  Show this help\n",
      argv[0]
    );
    return EXIT_FAILURE;
  };

  auto die = [](std::string_view msg) -> int {
    std::fprintf(stdout, "error: %.*s\n", int(msg.size()), msg.data());
    return EXIT_FAILURE;
  };

  auto require_value = [&](int& i, std::string_view opt) -> std::string_view {
    if (i + 1 >= argc) throw opt;
    return argv[++i];
  };

  auto parse_u32 = [&](std::string_view s, std::string_view opt) -> unsigned {
    unsigned v{};
    auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (ec != std::errc{} || p != s.data() + s.size()) {
      std::fprintf(
        stdout, "error: invalid value for %.*s: '%.*s'\n",
        int(opt.size()), opt.data(), int(s.size()), s.data()
      );
      throw opt;
    }
    return v;
  };

  auto parse_f32 = [&](std::string_view s, std::string_view opt) -> float {
    float v{};
    auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (ec != std::errc{} || p != s.data() + s.size()) {
      std::fprintf(
        stdout, "error: invalid value for %.*s: '%.*s'\n",
        int(opt.size()), opt.data(), int(s.size()), s.data()
      );
      throw opt;
    }
    return v;
  };

  std::function<std::vector<size_t>(size_t, size_t, float, float, float)> init_edges = logarithmic;
  unsigned fps = 30;
  unsigned width = 79;
  float min_freq = 20.0f;
  float max_freq = -1.0f; // <=0 => Nyquist
  float min_dbfs = -60.0f;
  float max_dbfs = 0.0f;

  bool use_jack = false;
  bool stereo = false;
  std::string filename;

  try {
    for (int i = 1; i < argc; ++i) {
      std::string_view a = argv[i];

      if (a == "--help") {
        return usage();
      } else if (a == "--fps") {
        fps = parse_u32(require_value(i, "--fps"), "--fps");
      } else if (a == "--width") {
        width = parse_u32(require_value(i, "--width"), "--width");
      } else if (a == "--min-freq") {
        auto freq = parse_freq(require_value(i, "--min-freq"));
        if (!freq) throw std::runtime_error("--min-freq: invalid argument");
        min_freq = *freq;
      } else if (a == "--max-freq") {
        auto freq = parse_freq(require_value(i, "--max-freq"));
        if (!freq) throw std::runtime_error("--max-freq: invalid argument");
        max_freq = *freq;
      } else if (a == "--min-dbfs") {
        min_dbfs = parse_f32(require_value(i, "--min-dbfs"), "--min-dbfs");
      } else if (a == "--max-dbfs") {
        max_dbfs = parse_f32(require_value(i, "--max-dbfs"), "--max-dbfs");
      } else if (a == "--linear") {
        init_edges = linear;
      } else if (a == "--cents-per-bin") {
        auto cents = parse_f32(require_value(i, "--cents-per-bin"), "--cents-per-bin");
        if (cents <= 0.0f) return die("--cents-per-bin must be > 0");
        init_edges = log_step{cents / 1200.0f};
      } else if (a == "--bins-per-octave") {
        auto bpo = parse_f32(require_value(i, "--bins-per-octave"), "--bins-per-octave");
        if (bpo <= 0.0f) return die("--bins-per-octave must be > 0");
        init_edges = log_step{1.0f / bpo};
      } else if (a == "--jack") {
        use_jack = true;
      } else if (a == "--stereo") {
        stereo = true;
      } else if (!a.empty() && a.front() == '-') {
        std::fprintf(stdout, "error: unknown option: %.*s\n", int(a.size()), a.data());
        return usage();
      } else {
        if (!filename.empty()) {
          std::fprintf(
            stdout, "error: multiple filenames given: '%s' and '%.*s'\n",
            filename.c_str(), int(a.size()), a.data()
          );
          return usage();
        }
        filename = std::string(a);
      }
    }
  } catch (std::string_view opt) {
    std::fprintf(stdout, "error: missing value for %.*s\n", int(opt.size()), opt.data());
    return usage();
  }

  if (use_jack && !filename.empty())
    return die("choose either --jack or <filename>, not both");
  if (!use_jack && filename.empty())
    return die("missing input source: specify --jack or <filename>");

  if (fps == 0)   return die("--fps must be > 0");
  if (width == 0) return die("--width must be > 0");

  std::unique_ptr<AudioSource> src;
  try {
    if (use_jack) src = std::make_unique<JACK>(stereo ? 2 : 1);
    else          src = std::make_unique<SoundFile>(filename);
  } catch (const std::exception& e) {
    std::fprintf(stdout, "error: failed to open input: %s\n", e.what());
    return EXIT_FAILURE;
  }

  if (stereo && src->channels() != 2)
    return die("--stereo requires exactly 2 channels");

  return waterfall(*src, init_edges, fps, width, min_dbfs, max_dbfs, min_freq, max_freq, stereo);
}
