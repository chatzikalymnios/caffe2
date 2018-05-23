#include "caffe2/core/context_gpu.h"

#include "caffe2/core/common_gpu.h"
#include "caffe2/core/cudnn_wrappers.h"
#include "caffe2/operators/conv_op.h"
#include "caffe2/operators/conv_op_cache_cudnn.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/operators/op_utils_cudnn.h"

#include "caffe2/utils/math.h"
#include <ctime>

namespace caffe2 {

namespace {

template <typename T>
__global__ void pad_and_lower_gpu_kernel_nhwc(
    const int size,      /* Total size of output tensor (L tensor) */
    const T* data_im,    /* Pointer to input tensor */
    const int channels,  /* Number of channels in input (C dimension) */
    const int height,    /* Height of input (H dimension) */
    const int padded_h,  /* Height of padded input (H dimension) */
    const int width,     /* Width of input (W dimension) */
    const int kernel_w,  /* Kernel width */
    const int pad_t,     /* Top padding in the H dimension (bottom padding assumed to be the same) */
    const int pad_l,     /* Left padding in the W dimension (right padding assumed to be the same) */
    const int stride_h,  /* Kernel stride in the H dimension */
    const int stride_w,  /* Kernel stride in the W dimension */
    const int width_out, /* Width of the convolution operation (eventual) output tensor (O tensor) */
    T* data_low          /* Pointer to the output tensor for the lowering operation (L tensor) */
    ) {

  int padded_height = height + 2 * pad_t;

  size_t index = blockIdx.x * blockDim.x + threadIdx.x;

  int H_factor = kernel_w * channels;
  int O_factor = padded_height * H_factor;
  int N_factor = width_out * O_factor;

  if (index < size) {
    int n = index / N_factor;
    int o_index = index % N_factor;
    int o = o_index / O_factor;
    int h_index = o_index % O_factor;
    int h = h_index / H_factor;
    int w_index = h_index % H_factor;
    int w = w_index / channels;
    int c = w_index % channels;

    T value = 0.0;

    int w_pos = w + o * stride_w;

    if ( (h >= pad_t) && (h < padded_height - pad_t) && (w_pos >= pad_l) && (w_pos < pad_l + width) ) {
      value = data_im[n * channels * height * width
                                    + (h - pad_t) * width * channels
                                    + (w_pos - pad_l) * channels
                                    + c];
    }

    data_low[index] = value;
  }
}

template <typename T>
__global__ void pad_and_lower_gpu_kernel_nchw(
    const int size,      /* Total size of output tensor (L tensor) */
    const T* data_im,    /* Pointer to input tensor */
    const int channels,  /* Number of channels in input (C dimension) */
    const int height,    /* Height of input (H dimension) */
    const int padded_h,  /* Height of padded input (H dimension) */
    const int width,     /* Width of input (W dimension) */
    const int kernel_w,  /* Kernel width */
    const int pad_t,     /* Top padding in the H dimension (bottom padding assumed to be the same) */
    const int pad_l,     /* Left padding in the W dimension (right padding assumed to be the same) */
    const int stride_h,  /* Kernel stride in the H dimension */
    const int stride_w,  /* Kernel stride in the W dimension */
    const int width_out, /* Width of the convolution operation (eventual) output tensor (O tensor) */
    T* data_low          /* Pointer to the output tensor for the lowering operation (L tensor) */
    ) {

  size_t index = blockIdx.x * blockDim.x + threadIdx.x;

  int H_factor = kernel_w * channels;
  int O_factor = padded_h * H_factor;
  int N_factor = width_out * O_factor;

  if (index < size) {
    int n = index / N_factor;
    int o_index = index % N_factor;
    int o = o_index / O_factor;
    int h_index = o_index % O_factor;
    int h = h_index / H_factor;
    int w_index = h_index % H_factor;
    int w = w_index / channels;
    int c = w_index % channels;

    T value = 0.0;

    int w_pos = w + o * stride_w;

    if ( (h >= pad_t) && (h < padded_h - pad_t) && (w_pos >= pad_l) && (w_pos < pad_l + width) ) {
      value = data_im[n * channels * height * width
                                    + c * height * width
                                    + (h - pad_t) * width
                                    + (w_pos - pad_l)];
    }

    data_low[index] = value;
  }
}

template <typename T>
__global__ void impr_pad_and_lower_gpu_kernel_nhwc(
    const int size,      /* Total size of output tensor (L tensor) */
    const T* data_im,    /* Pointer to input tensor */
    const int channels,  /* Number of channels in input (C dimension) */
    const int height,    /* Height of input (H dimension) */
    const int padded_h,  /* Height of padded input (H dimension) */
    const int width,     /* Width of input (W dimension) */
    const int kernel_w,  /* Kernel width */
    const int pad_t,     /* Top padding in the H dimension (bottom padding assumed to be the same) */
    const int pad_l,     /* Left padding in the W dimension (right padding assumed to be the same) */
    const int stride_h,  /* Kernel stride in the H dimension */
    const int stride_w,  /* Kernel stride in the W dimension */
    const int width_out, /* Width of the convolution operation (eventual) output tensor (O tensor) */
    T* data_low          /* Pointer to the output tensor for the lowering operation (L tensor) */
    ) {

  int H_factor = kernel_w * channels;
  int O_factor = padded_h * H_factor;
  int N_factor = width_out * O_factor;

  int N_idx = blockIdx.x;
  int O_idx = blockIdx.y;

  int tid = blockIdx.z * blockDim.x + threadIdx.x;

  int H_idx = tid / H_factor;
  int rem = tid % H_factor;
  int W_idx = rem / channels;
  int C_idx = rem % channels;

  int idx = N_idx * N_factor + O_idx * O_factor + tid;

  if (tid < O_factor && idx < size) {
    T value = 0.0;

    int w_pos = W_idx + O_idx * stride_w;

    if ( (H_idx >= pad_t) && (H_idx < padded_h - pad_t) && (w_pos >= pad_l) && (w_pos < pad_l + width) ) {
      value = data_im[N_idx * channels * height * width
                                    + (H_idx - pad_t) * width * channels
                                    + (w_pos - pad_l) * channels
                                    + C_idx];
    }

    data_low[idx] = value;
  }
}

template <typename T>
__global__ void impr_pad_and_lower_gpu_kernel_nchw(
    const int size,      /* Total size of output tensor (L tensor) */
    const T* data_im,    /* Pointer to input tensor */
    const int channels,  /* Number of channels in input (C dimension) */
    const int height,    /* Height of input (H dimension) */
    const int padded_h,  /* Height of padded input (H dimension) */
    const int width,     /* Width of input (W dimension) */
    const int kernel_w,  /* Kernel width */
    const int pad_t,     /* Top padding in the H dimension (bottom padding assumed to be the same) */
    const int pad_l,     /* Left padding in the W dimension (right padding assumed to be the same) */
    const int stride_h,  /* Kernel stride in the H dimension */
    const int stride_w,  /* Kernel stride in the W dimension */
    const int width_out, /* Width of the convolution operation (eventual) output tensor (O tensor) */
    T* data_low          /* Pointer to the output tensor for the lowering operation (L tensor) */
    ) {

  int H_factor = kernel_w * channels;
  int O_factor = padded_h * H_factor;
  int N_factor = width_out * O_factor;

  int N_idx = blockIdx.x;
  int O_idx = blockIdx.y;

  int tid = blockIdx.z * blockDim.x + threadIdx.x;

  int H_idx = tid / H_factor;
  int rem = tid % H_factor;
  int W_idx = rem / channels;
  int C_idx = rem % channels;

  int idx = N_idx * N_factor + O_idx * O_factor + tid;

  if (tid < O_factor && idx < size) {
    T value = 0.0;

    int w_pos = W_idx + O_idx * stride_w;

    if ( (H_idx >= pad_t) && (H_idx < padded_h - pad_t) && (w_pos >= pad_l) && (w_pos < pad_l + width) ) {
      value = data_im[N_idx * channels * height * width
                                    + C_idx * height * width
                                    + (H_idx - pad_t) * width
                                    + (w_pos - pad_l)];
    }

    data_low[idx] = value;
  }
}

} // namespace

class MecOpBase : public ConvPoolOpBase<CUDAContext> {
 public:
  MecOpBase(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CUDAContext>(operator_def, ws),
        cudnn_wrapper_(&context_),
        cudnn_state_(OperatorBase::GetSingleArgument<int>("cudnn_state", 0)) {
    for (int i = 0; i < kernel_.size(); ++i) {
      OPERATOR_NEEDS_FEATURE(
          pads_[i] == pads_[kernel_.size() + i],
          "The current padding scheme leads to unequal padding on the left "
          "and right, which is not supported by cudnn.");
    }

    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&filter_desc_));
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&bias_desc_));

    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&K_desc_));
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&O_desc_));

    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&top_desc_));
  }

  ~MecOpBase() {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(filter_desc_));
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(bias_desc_));

    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(K_desc_));
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(O_desc_));

    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(top_desc_));

//    cudaFree(d_pointers);
  }

 protected:
  // A helper function to set up the tensor Nd desriptor, depending on the order
  // the group and the type given.
  template <typename T>
  void SetTensorNdDescriptor(
      int size,
      StorageOrder order,
      cudnnTensorDescriptor_t tensorDesc,
      int N,
      int C,
      int H,
      int W,
      int D) {
    switch (order) {
      case StorageOrder::NHWC:
        if (size == 4) {
          CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(
              tensorDesc,
              cudnnTypeWrapper<T>::type,
              N,
              C,
              H,
              W,
              H * W * C,
              1,
              W * C,
              C));
        } else {
          vector<int> dims = {N, H, W, D, C};
          vector<int> strides = {H * W * D * C, W * D * C, D * C, C, 1};
          CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
              tensorDesc,
              cudnnTypeWrapper<T>::type,
              size > 3 ? size : 4,
              dims.data(),
              strides.data()));
        }
        break;
      case StorageOrder::NCHW:
        if (size == 4) {
          CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(
              tensorDesc,
              cudnnTypeWrapper<T>::type,
              N,
              C,
              H,
              W,
              C * H * W,
              H * W,
              W,
              1));
        } else {
          vector<int> dims = {N, C, H, W, D};
          vector<int> strides = {C * H * W * D, H * W * D, W * D, D, 1};
          CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
              tensorDesc,
              cudnnTypeWrapper<T>::type,
              size > 3 ? size : 4,
              dims.data(),
              strides.data()));
        }
        break;
      default:
        LOG(FATAL) << "Unknown storage order: " << order_;
    }
  }

  //vector<TIndex> cudnn_input_dims_;
  //vector<TIndex> cudnn_filter_dims_;

  CuDNNWrapper cudnn_wrapper_;

  cudnnTensorDescriptor_t filter_desc_; // W
  cudnnTensorDescriptor_t bias_desc_;   // b

  // Intermediate tensors
  cudnnTensorDescriptor_t K_desc_; // K in MEC paper
  cudnnTensorDescriptor_t O_desc_; // O in MEC paper

  cudnnTensorDescriptor_t top_desc_;    // Y

  const float **d_pointers;

  size_t cudnn_state_;
};

class MecOp final : public MecOpBase {
 public:
  MecOp(const OperatorDef& operator_def, Workspace* ws)
      : MecOpBase(operator_def, ws) {}

  ~MecOp() {}

  template <typename T_X, typename T_W, typename T_B, typename T_Y>
  bool Solution_A_DoRunWithType();

  template <typename T_X, typename T_W, typename T_B, typename T_Y>
  bool Solution_B_DoRunWithType();

  bool RunOnDevice() override;

 private:
  // Input: X, W, b
  // Output: Y
  INPUT_TAGS(INPUT, FILTER, BIAS);
};

////////////////////////////////////////////////////////////////////////////////
// Implementations
////////////////////////////////////////////////////////////////////////////////

template <typename T_X, typename T_W, typename T_B, typename T_Y>
bool MecOp::Solution_A_DoRunWithType() {
  auto& X = Input(INPUT);
  auto& filter = Input(FILTER);
  auto* Y = Output(0);

//  clock_t begin = clock();

  // Figure out the output shape
  CAFFE_ENFORCE(X.ndim() >= 3 && X.ndim() <= 5);
  CAFFE_ENFORCE(filter.ndim() >= 3 && filter.ndim() <= 5);
  const int M = filter.dim32(0);
  ConvPoolOpBase<CUDAContext>::SetOutputSize(X, Y, M);
  int N = 0, C = 0, H = 0, W = 0, D = 0, H_out = 0, W_out = 0, D_out = 0;

  int I_n = 0, I_c = 0, I_h = 0, I_w = 0, K_h = 0, K_w = 0, K_c = 0;

  Tensor<CUDAContext> *O;

  switch (order_) {
    case StorageOrder::NHWC:
      N = X.dim32(0);
      H = X.dim32(1);
      W = X.ndim() > 3 ? X.dim32(2) : 1;
      D = X.ndim() > 4 ? X.dim32(3) : 1;
      C = X.dim32(X.ndim() - 1);
      H_out = Y->dim32(1);
      W_out = Y->ndim() > 3 ? Y->dim32(2) : 1;
      D_out = Y->ndim() > 4 ? Y->dim32(3) : 1;
      for (int i = 0; i < kernel_.size(); ++i) {
        CAFFE_ENFORCE_EQ(filter.dim32(i + 1), kernel_[i]);
      }
      CAFFE_ENFORCE_EQ(filter.dim32(filter.ndim() - 1), C);

      I_n = N;
      I_h = H + pad_t() + pad_b();
      I_w = W + pad_l() + pad_r();
      I_c = C;

      K_h = filter.dim32(1);
      K_w = filter.dim32(2);
      K_c = filter.dim32(0);

      // Descriptors

      CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(
        filter_desc_,
        cudnnTypeWrapper<T_W>::type,
        K_c, C, K_h, K_w,
        C * K_h * K_w, 1, K_w * C, C)
      );

      CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(
        top_desc_,
        cudnnTypeWrapper<T_Y>::type,
        I_n, K_c, H_out, W_out,
        K_c * H_out * W_out, 1, W_out * K_c, K_c)
      );

      O = new Tensor<CUDAContext>((vector<int>) { I_n, H_out, W_out * K_c });
      CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(
        O_desc_,
        cudnnTypeWrapper<T_Y>::type,
        I_n, K_c, H_out, W_out,
        K_c * W_out, 1, I_n * W_out * K_c, K_c)
      );

      break;
    case StorageOrder::NCHW:
      N = X.dim32(0);
      C = X.dim32(1);
      H = X.dim32(2);
      W = X.ndim() > 3 ? X.dim32(3) : 1;
      D = X.ndim() > 4 ? X.dim32(4) : 1;
      H_out = Y->dim32(2);
      W_out = Y->ndim() > 3 ? Y->dim32(3) : 1;
      D_out = Y->ndim() > 4 ? Y->dim32(4) : 1;
      CAFFE_ENFORCE_EQ(filter.dim32(1), C);
      for (int i = 0; i < kernel_.size(); ++i) {
        CAFFE_ENFORCE_EQ(filter.dim32(i + 2), kernel_[i]);
      }

      I_n = N;
      I_h = H + pad_t() + pad_b();
      I_w = W + pad_l() + pad_r();
      I_c = C;

      K_h = filter.dim32(2);
      K_w = filter.dim32(3);
      K_c = filter.dim32(0);

      // Descriptors
      CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(
        filter_desc_,
        cudnnTypeWrapper<T_W>::type,
        K_c, C, K_h, K_w,
        C * K_h * K_w, K_h * K_w, K_w, 1)
      );

      CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(
        top_desc_,
        cudnnTypeWrapper<T_Y>::type,
        N, K_c, H_out, W_out,
        K_c * H_out * W_out, H_out * W_out, W_out, 1)
      );

      O = new Tensor<CUDAContext>((vector<int>) { I_n, H_out, W_out * K_c });
      CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(
        O_desc_,
        cudnnTypeWrapper<T_Y>::type,
        I_n, K_c, H_out, W_out,
        K_c * W_out, 1, I_n * W_out * K_c, K_c)
      );

      break;
    default:
      LOG(FATAL) << "Unknown storage order: " << order_;
  }

//  cudaDeviceSynchronize();
//  clock_t init_end = clock();

  // Convert filter
  Tensor<CUDAContext> K ((vector<int>) { K_h, K_w, I_c, K_c });
  CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(
    K_desc_,
    cudnnTypeWrapper<T_W>::type,
    K_c, C, K_h, K_w,
    1, K_c, K_w * C * K_c, C * K_c)
  );

  cudnn_wrapper_.with_cudnn_state(cudnn_state_, [&](CuDNNState* state) {
    CUDNN_ENFORCE(cudnnTransformTensor(
      state->cudnn_handle(),
      cudnnTypeWrapper<T_W>::kOne(),
      filter_desc_,
      filter.template data<T_W>(),
      cudnnTypeWrapper<T_W>::kZero(),
      K_desc_,
      K.template mutable_data<T_W>()));
  });

//  cudaDeviceSynchronize();
//  clock_t filc_end = clock();

  // Pad and lower input

  Tensor<CUDAContext> L((vector<int>) { I_n, W_out, I_h, K_w * I_c });

  const float* X_data = X.template data<float>();
  float* L_data = L.template mutable_data<float>();

  dim3 grid(I_n, W_out, CAFFE_GET_BLOCKS(I_h * K_w * I_c));

  if (order_ == StorageOrder::NCHW) {
    impr_pad_and_lower_gpu_kernel_nchw<float><<<
      grid,
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
        (int) L.size(),
        X_data,
        C,
        H,
        I_h,
        W,
        K_w,
        pad_t(),
        pad_l(),
        stride_h(),
        stride_w(),
        W_out,
        L_data
    );
  } else {
    impr_pad_and_lower_gpu_kernel_nhwc<float><<<
      grid,
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
        (int) L.size(),
        X_data,
        C,
        H,
        I_h,
        W,
        K_w,
        pad_t(),
        pad_l(),
        stride_h(),
        stride_w(),
        W_out,
        L_data
    );
  }

//  TensorPrinter tp;
//
//  Tensor<CPUContext> CPU(X);
//  tp.Print<float>(CPU);
//
//  Tensor<CPUContext> CPU2(L);
//  tp.Print<float>(CPU2);

//  cudaDeviceSynchronize();
//  clock_t lower_end = clock();

  // SOLUTION A

  const float *K_data = K.template data<float>();
  float *O_data = O->template mutable_data<float>();

  const float Alpha = 1.0f;
  const float Beta = 0.0f;

    CUBLAS_ENFORCE(cublasSgemmStridedBatched(
        context_.cublas_handle(),
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        K_c,
        I_n * W_out,
        K_h * K_w * I_c,
        &Alpha,
        K_data,
        K_c,
        0,
        L_data,
        I_h * K_w * I_c,
        stride_h() * K_w * I_c,
        &Beta,
        O_data,
        K_c,
        I_n * W_out * K_c,
        H_out));

  cudnn_wrapper_.with_cudnn_state(cudnn_state_, [&](CuDNNState* state) {
    CUDNN_ENFORCE(cudnnTransformTensor(
      state->cudnn_handle(),
      cudnnTypeWrapper<T_Y>::kOne(),
      O_desc_,
      O->template data<T_Y>(),
      cudnnTypeWrapper<T_Y>::kZero(),
      top_desc_,
      Y->template mutable_data<T_Y>()));
  });

  delete(O);

//  cudaDeviceSynchronize();
//  clock_t trans_end = clock();

  // Bias
  if (InputSize() == 3) {
    auto& bias = Input(BIAS);

    CAFFE_ENFORCE_EQ(bias.ndim(), 1);
    CAFFE_ENFORCE_EQ(bias.dim32(0), M);

    CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(
      bias_desc_,
      cudnnTypeWrapper<T_B>::type,
      1, K_c, 1, 1,
      K_c, 1, 1, 1)
    );

    CUDNN_ENFORCE(cudnnAddTensor(
        cudnn_wrapper_.inline_cudnn_handle(),
        cudnnTypeWrapper<T_B>::kOne(),
        bias_desc_,
        bias.template data<T_B>(),
        cudnnTypeWrapper<T_Y>::kOne(),
        top_desc_,
        Y->template mutable_data<T_Y>()));
  }

//  cudaDeviceSynchronize();
//  clock_t end = clock();
//
//  static int conv_num = 0;
//  conv_num++;
//  double elapsed_ms = double(end - begin) / CLOCKS_PER_SEC * 1000;
//
//  LOG(INFO) << "CONV " << conv_num << "\n\t" <<
//    "N: " << N << " C: " << C << " H: " << H << " W: " << W << " K_c: " << K_c << " K_h: " << K_h << " K_w: " << K_w << "\n\t" <<
//    "stride: " << stride_h() << " pad: " << pad_t() << "\n\t" <<
//    "TIME: " << elapsed_ms << " ms\n\t" <<
//    "INIT: " << double(init_end - begin) / CLOCKS_PER_SEC * 1000 << " ms" << "\n\t" <<
//    "FILC: " << double(filc_end - init_end) / CLOCKS_PER_SEC * 1000 << " ms" << "\n\t" <<
//    "LOWE: " << double(lower_end - filc_end) / CLOCKS_PER_SEC * 1000 << " ms" << "\n\t" <<
//    "GEMM: " << double(gemm_end - lower_end) / CLOCKS_PER_SEC * 1000 << " ms" << "\n\t" <<
//    "TRAN: " << double(trans_end - gemm_end) / CLOCKS_PER_SEC * 1000 << " ms" << "\n\t" <<
//    "BIAS: " << double(end - trans_end) / CLOCKS_PER_SEC * 1000 << " ms";

  return true;
}

template <typename T_X, typename T_W, typename T_B, typename T_Y>
bool MecOp::Solution_B_DoRunWithType() {
  auto& X = Input(INPUT);
  auto& filter = Input(FILTER);
  auto* Y = Output(0);

//  clock_t begin = clock();

  // Figure out the output shape
  CAFFE_ENFORCE(X.ndim() >= 3 && X.ndim() <= 5);
  CAFFE_ENFORCE(filter.ndim() >= 3 && filter.ndim() <= 5);
  const int M = filter.dim32(0);
  ConvPoolOpBase<CUDAContext>::SetOutputSize(X, Y, M);
  int N = 0, C = 0, H = 0, W = 0, D = 0, H_out = 0, W_out = 0, D_out = 0;

  int I_n = 0, I_c = 0, I_h = 0, I_w = 0, K_h = 0, K_w = 0, K_c = 0;

  Tensor<CUDAContext> *O;

  switch (order_) {
    case StorageOrder::NHWC:
      N = X.dim32(0);
      H = X.dim32(1);
      W = X.ndim() > 3 ? X.dim32(2) : 1;
      D = X.ndim() > 4 ? X.dim32(3) : 1;
      C = X.dim32(X.ndim() - 1);
      H_out = Y->dim32(1);
      W_out = Y->ndim() > 3 ? Y->dim32(2) : 1;
      D_out = Y->ndim() > 4 ? Y->dim32(3) : 1;
      for (int i = 0; i < kernel_.size(); ++i) {
        CAFFE_ENFORCE_EQ(filter.dim32(i + 1), kernel_[i]);
      }
      CAFFE_ENFORCE_EQ(filter.dim32(filter.ndim() - 1), C);

      I_n = N;
      I_h = H + pad_t() + pad_b();
      I_w = W + pad_l() + pad_r();
      I_c = C;

      K_h = filter.dim32(1);
      K_w = filter.dim32(2);
      K_c = filter.dim32(0);

      // Descriptors

      CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(
        filter_desc_,
        cudnnTypeWrapper<T_W>::type,
        K_c, C, K_h, K_w,
        C * K_h * K_w, 1, K_w * C, C)
      );

      CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(
        top_desc_,
        cudnnTypeWrapper<T_Y>::type,
        I_n, K_c, H_out, W_out,
        K_c * H_out * W_out, 1, W_out * K_c, K_c)
      );

      O = Y;

      break;
    case StorageOrder::NCHW:
      N = X.dim32(0);
      C = X.dim32(1);
      H = X.dim32(2);
      W = X.ndim() > 3 ? X.dim32(3) : 1;
      D = X.ndim() > 4 ? X.dim32(4) : 1;
      H_out = Y->dim32(2);
      W_out = Y->ndim() > 3 ? Y->dim32(3) : 1;
      D_out = Y->ndim() > 4 ? Y->dim32(4) : 1;
      CAFFE_ENFORCE_EQ(filter.dim32(1), C);
      for (int i = 0; i < kernel_.size(); ++i) {
        CAFFE_ENFORCE_EQ(filter.dim32(i + 2), kernel_[i]);
      }

      I_n = N;
      I_h = H + pad_t() + pad_b();
      I_w = W + pad_l() + pad_r();
      I_c = C;

      K_h = filter.dim32(2);
      K_w = filter.dim32(3);
      K_c = filter.dim32(0);

      // Descriptors
      CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(
        filter_desc_,
        cudnnTypeWrapper<T_W>::type,
        K_c, C, K_h, K_w,
        C * K_h * K_w, K_h * K_w, K_w, 1)
      );

      CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(
        top_desc_,
        cudnnTypeWrapper<T_Y>::type,
        N, K_c, H_out, W_out,
        K_c * H_out * W_out, H_out * W_out, W_out, 1)
      );

      O = new Tensor<CUDAContext>((vector<int>) { I_n, H_out, W_out * K_c });
      CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(
        O_desc_,
        cudnnTypeWrapper<T_Y>::type,
        I_n, K_c, H_out, W_out,
        K_c * H_out * W_out, 1, W_out * K_c, K_c)
      );

      break;
    default:
      LOG(FATAL) << "Unknown storage order: " << order_;
  }

//  cudaDeviceSynchronize();
//  clock_t init_end = clock();

  // Convert filter
  Tensor<CUDAContext> K ((vector<int>) { K_h, K_w, I_c, K_c });
  CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(
    K_desc_,
    cudnnTypeWrapper<T_W>::type,
    K_c, C, K_h, K_w,
    1, K_c, K_w * C * K_c, C * K_c)
  );

  cudnn_wrapper_.with_cudnn_state(cudnn_state_, [&](CuDNNState* state) {
    CUDNN_ENFORCE(cudnnTransformTensor(
      state->cudnn_handle(),
      cudnnTypeWrapper<T_W>::kOne(),
      filter_desc_,
      filter.template data<T_W>(),
      cudnnTypeWrapper<T_W>::kZero(),
      K_desc_,
      K.template mutable_data<T_W>()));
  });

//  cudaDeviceSynchronize();
//  clock_t filc_end = clock();

  // Pad and lower input

  Tensor<CUDAContext> L((vector<int>) { I_n, W_out, I_h, K_w * I_c });

  const float* X_data = X.template data<float>();
  float* L_data = L.template mutable_data<float>();

  dim3 grid(I_n, W_out, CAFFE_GET_BLOCKS(I_h * K_w * I_c));

  if (order_ == StorageOrder::NCHW) {
    impr_pad_and_lower_gpu_kernel_nchw<float><<<
      grid,
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
        (int) L.size(),
        X_data,
        C,
        H,
        I_h,
        W,
        K_w,
        pad_t(),
        pad_l(),
        stride_h(),
        stride_w(),
        W_out,
        L_data
    );
  } else {
    impr_pad_and_lower_gpu_kernel_nhwc<float><<<
      grid,
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
        (int) L.size(),
        X_data,
        C,
        H,
        I_h,
        W,
        K_w,
        pad_t(),
        pad_l(),
        stride_h(),
        stride_w(),
        W_out,
        L_data
    );
  }

//  TensorPrinter tp;
//
//  Tensor<CPUContext> CPU(X);
//  tp.Print<float>(CPU);
//
//  Tensor<CPUContext> CPU2(L);
//  tp.Print<float>(CPU2);

//  cudaDeviceSynchronize();
//  clock_t lower_end = clock();

  // SOLUTION B

  const float *K_data = K.template data<float>();
  float *O_data = O->template mutable_data<float>();

  const float Alpha = 1.0f;
  const float Beta = 0.0f;

  int batch_size = I_n * H_out;

  vector<const float *> pointers(3 * batch_size);

  int L_0 = W_out * I_h * K_w * I_c;
  int L_2 = stride_h() * K_w * I_c;

  int O_0 = H_out * W_out * K_c;
  int O_1 = W_out * K_c;

  int counter = 0;

  for (int n = 0; n < I_n; n++) {
    for (int h = 0; h < H_out; h++) {
        int L_offset = n * L_0 + h * L_2;
        int O_offset = n * O_0 + h * O_1;

        pointers[counter] = K_data;
        pointers[batch_size + counter] = L_data + L_offset;
        pointers[2 * batch_size + counter] = O_data + O_offset;
        counter++;
    }
  }

  // Memory freed in destructor
  cudaMalloc((void**) &d_pointers, 3 * batch_size * sizeof(float *));
  cudaMemcpy(d_pointers, pointers.data(), 3 * batch_size * sizeof(float *), cudaMemcpyHostToDevice);

  CUBLAS_ENFORCE(cublasSgemmBatched(
        context_.cublas_handle(),
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        K_c,
        W_out,
        K_h * K_w * I_c,
        &Alpha,
        d_pointers,
        K_c,
        d_pointers + batch_size,
        I_h * K_w * I_c,
        &Beta,
        (float **) (d_pointers + 2 * batch_size),
        K_c,
        batch_size));

//  cudaDeviceSynchronize();
//  clock_t gemm_end = clock();

  if (order_ == StorageOrder::NCHW) {
    cudnn_wrapper_.with_cudnn_state(cudnn_state_, [&](CuDNNState* state) {
      CUDNN_ENFORCE(cudnnTransformTensor(
        state->cudnn_handle(),
        cudnnTypeWrapper<T_Y>::kOne(),
        O_desc_,
        O->template data<T_Y>(),
        cudnnTypeWrapper<T_Y>::kZero(),
        top_desc_,
        Y->template mutable_data<T_Y>()));
    });

    delete(O);
  }

//  cudaDeviceSynchronize();
//  clock_t trans_end = clock();

  // Bias
  if (InputSize() == 3) {
    auto& bias = Input(BIAS);

    CAFFE_ENFORCE_EQ(bias.ndim(), 1);
    CAFFE_ENFORCE_EQ(bias.dim32(0), M);

    CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(
      bias_desc_,
      cudnnTypeWrapper<T_B>::type,
      1, K_c, 1, 1,
      K_c, 1, 1, 1)
    );

    CUDNN_ENFORCE(cudnnAddTensor(
        cudnn_wrapper_.inline_cudnn_handle(),
        cudnnTypeWrapper<T_B>::kOne(),
        bias_desc_,
        bias.template data<T_B>(),
        cudnnTypeWrapper<T_Y>::kOne(),
        top_desc_,
        Y->template mutable_data<T_Y>()));
  }

//  cudaDeviceSynchronize();
//  clock_t end = clock();
//
//  static int conv_num = 0;
//  conv_num++;
//  double elapsed_ms = double(end - begin) / CLOCKS_PER_SEC * 1000;
//
//  LOG(INFO) << "CONV " << conv_num << "\n\t" <<
//    "N: " << N << " C: " << C << " H: " << H << " W: " << W << " K_c: " << K_c << " K_h: " << K_h << " K_w: " << K_w << "\n\t" <<
//    "stride: " << stride_h() << " pad: " << pad_t() << "\n\t" <<
//    "TIME: " << elapsed_ms << " ms\n\t" <<
//    "INIT: " << double(init_end - begin) / CLOCKS_PER_SEC * 1000 << " ms" << "\n\t" <<
//    "FILC: " << double(filc_end - init_end) / CLOCKS_PER_SEC * 1000 << " ms" << "\n\t" <<
//    "LOWE: " << double(lower_end - filc_end) / CLOCKS_PER_SEC * 1000 << " ms" << "\n\t" <<
//    "GEMM: " << double(gemm_end - lower_end) / CLOCKS_PER_SEC * 1000 << " ms" << "\n\t" <<
//    "TRAN: " << double(trans_end - gemm_end) / CLOCKS_PER_SEC * 1000 << " ms" << "\n\t" <<
//    "BIAS: " << double(end - trans_end) / CLOCKS_PER_SEC * 1000 << " ms";

  return true;
}

bool MecOp::RunOnDevice() {
  if (Input(0).IsType<float>()) {
    return Solution_B_DoRunWithType<
        float, // X
        float, // W
        float, // B
        float>(); // Y
  } else if (Input(0).IsType<float16>()) {
    return Solution_B_DoRunWithType<
        float16, // X
        float16, // W
        float16, // B
        float16>(); // Y
  } else {
    LOG(FATAL) << "Only float (32bit) and float16 are supported by "
               << "cudnn convolution, but input " << debug_def().input(0)
               << " has [" << Input(0).meta().name() << "]";
  }
  return true;
}

REGISTER_CUDNN_OPERATOR(Conv_MEC, MecOp);

} // namespace caffe2
