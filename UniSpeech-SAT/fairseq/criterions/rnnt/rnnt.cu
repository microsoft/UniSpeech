#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>


// TODO: add some template for int type as well


// Calculate gradient for last blank that has not Beta factor. Also division by
// log_probs_a.size(0) is applied here, instead of in Python code
template <typename scalar_t>
__global__ void fill_grad_last_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> log_probs_a,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> alphas_a,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> betas_a,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> input_lengths_a,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> label_lengths_a,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> costs_a,
    int blank,
    float loss_scale) {

    const int mb = blockIdx.x * blockDim.x + threadIdx.x;

    if (mb < log_probs_a.size(0)) {
        int T = input_lengths_a[mb];
        int U = label_lengths_a[mb] + 1;

        log_probs_a[mb][T - 1][U - 1][blank] = (exp(alphas_a[mb][T - 1][U - 1] + betas_a[mb][T - 1][U - 1] +
                                                    log_probs_a[mb][T - 1][U - 1][blank] + costs_a[mb]) - 1) * loss_scale ;
    }
}

// Calculate gradient for blank
template <typename scalar_t>
__global__ void fill_grad_blank_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> log_probs_a,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> alphas_a,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> betas_a,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> input_lengths_a,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> label_lengths_a,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> costs_a,
    int blank, float loss_scale) {

    const int mb = blockIdx.z * blockDim.z + threadIdx.z;
    const int t = blockIdx.y * blockDim.y + threadIdx.y;
    const int u = blockIdx.x * blockDim.x + threadIdx.x;

    if (mb < log_probs_a.size(0)) {
        int T = input_lengths_a[mb];
        int U = label_lengths_a[mb] + 1;
        if (t < T - 1 && u < U) {
            log_probs_a[mb][t][u][blank] = (exp(alphas_a[mb][t][u] + betas_a[mb][t][u]  + log_probs_a[mb][t][u][blank] + costs_a[mb]) -
                                            exp(alphas_a[mb][t][u] + betas_a[mb][t + 1][u]  + log_probs_a[mb][t][u][blank] + costs_a[mb])) * loss_scale ;
        }
    }
}

// Calculate gradients for labels
template <typename scalar_t>
__global__ void fill_grad_label_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> log_probs_a,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> alphas_a,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> betas_a,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> labels_a,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> input_lengths_a,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> label_lengths_a,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> costs_a,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> cum_a,
    int blank, float loss_scale) {

    const int mb = blockIdx.z * blockDim.z + threadIdx.z;
    const int t = blockIdx.y * blockDim.y + threadIdx.y;
    const int u = blockIdx.x * blockDim.x + threadIdx.x;

    if (mb < log_probs_a.size(0)) {
        int T = input_lengths_a[mb];
        int U = label_lengths_a[mb] + 1;
        if (t < T && u < U - 1) {
            int l = labels_a[cum_a[mb] + u];
            log_probs_a[mb][t][u][l] = (exp(alphas_a[mb][t][u] + betas_a[mb][t][u] + log_probs_a[mb][t][u][l] + costs_a[mb]) -
                                        exp(alphas_a[mb][t][u] + betas_a[mb][t][u + 1] + log_probs_a[mb][t][u][l] + costs_a[mb])) * loss_scale ;
        }
    }
}

// In order to apply the memory efficient RNNT gradient update formula,
// except for labels and blank, zero also needs to be considered
template <typename scalar_t>
__global__ void fill_grad_zeros(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> log_probs_a,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> alphas_a,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> betas_a,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> labels_a,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> input_lengths_a,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> label_lengths_a,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> costs_a,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> cum_a,
    int blank, float loss_scale) {

    const int mb = blockIdx.z * blockDim.z + threadIdx.z;
    const int t = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int vocab_size = log_probs_a.size(3);
    const int u = k / vocab_size;
    const int v = k % vocab_size;

    if (mb < log_probs_a.size(0) && t < log_probs_a.size(1) &&
        u < log_probs_a.size(2) && v < log_probs_a.size(3)) {
        int T = input_lengths_a[mb];
        int U = label_lengths_a[mb] + 1;
        float dl_ds = 0;
       // TODO: simplify if-s here to speed up computation
        if (t < T && u < U - 1) {
            int l = labels_a[cum_a[mb] + u];
            if (v == l) {
                return;
            }
            dl_ds += -exp(alphas_a[mb][t][u] + betas_a[mb][t][u + 1] +
                          costs_a[mb] + log_probs_a[mb][t][u][l]);
        }
        if (t < T - 1 && u < U) {
            if (v == blank) {
                return;
            }
            dl_ds += -exp(alphas_a[mb][t][u] + betas_a[mb][t + 1][u] +
                          costs_a[mb] + log_probs_a[mb][t][u][blank]);
        }
        if (t == T - 1 && u == U - 1) {
            if (v == blank) {
                return;
            }
            dl_ds += -exp(alphas_a[mb][t][u] + costs_a[mb] +
                          log_probs_a[mb][t][u][blank]);
        }
        log_probs_a[mb][t][u][v] = -exp(log_probs_a[mb][t][u][v]) * dl_ds * loss_scale;
    }
}

// TODO: maybe if we keep g_blank and g_l as separate arrays memory access will be better

void compute_grads_cuda(
    torch::Tensor log_probs,
    torch::Tensor alphas,
    torch::Tensor betas,
    torch::Tensor labels,
    torch::Tensor input_lengths,
    torch::Tensor label_lengths,
    torch::Tensor costs,
    torch::Tensor cum,
    int blank,
    float loss_scale) {

    const int batch_size = log_probs.size(0);
    const int max_t = log_probs.size(1);
    const int max_u = log_probs.size(2);
    const int vocab_size = log_probs.size(3);

    const dim3 threads3(1024, 1, 1);
    const dim3 blocks3((max_u * vocab_size + threads3.x - 1) / threads3.x,
                       (max_t + threads3.y - 1) / threads3.y,
                       (batch_size + threads3.z - 1) / threads3.z);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(log_probs.scalar_type(), "compute_grads_fill_zeros", ([&] {
        fill_grad_zeros<scalar_t><<<blocks3, threads3>>>(
            log_probs.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            alphas.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            betas.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            labels.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            input_lengths.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            label_lengths.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            costs.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            cum.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            blank, loss_scale);
    }));

    cudaDeviceSynchronize();

    // TODO: maybe add a check for this
    // assuming batch_size is always smaller than number of threads!
    const int threads1 = batch_size;
    const int blocks1 = 1;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(log_probs.scalar_type(), "compute_grads_cuda_last", ([&] {
        fill_grad_last_kernel<scalar_t><<<blocks1, threads1>>>(
            log_probs.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            alphas.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            betas.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            input_lengths.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            label_lengths.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            costs.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            blank, loss_scale);
    }));

    // TODO: is this a good way to divide computation?
    // TODO: how many threads in total do we need?
    const dim3 threads2(32, 32, 1);
    const dim3 blocks2((max_u + threads2.x - 1) / threads2.x,
                       (max_t + threads2.y - 1) / threads2.y,
                       (batch_size + threads2.z - 1) / threads2.z);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(log_probs.scalar_type(), "compute_grads_cuda_blank", ([&] {
        fill_grad_blank_kernel<scalar_t><<<blocks2, threads2>>>(
            log_probs.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            alphas.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            betas.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            input_lengths.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            label_lengths.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            costs.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            blank, loss_scale);
    }));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(log_probs.scalar_type(), "compute_grads_cuda_label", ([&] {
        fill_grad_label_kernel<scalar_t><<<blocks2, threads2>>>(
            log_probs.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            alphas.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            betas.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            labels.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            input_lengths.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            label_lengths.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            costs.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            cum.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            blank, loss_scale);
    }));

    cudaDeviceSynchronize();
}

// calculate log_sum because all calcuation is in log space that avoids the precision loss
// template <typename scalar_t>
__device__ __forceinline__ float log_sum_exp(float a, float b) {

    if (!isfinite(a)) return b;
    if (!isfinite(b)) return a;
    if (a > b)
        return log1p(exp(b - a)) + a;
    else
        return log1p(exp(a - b)) + b;
}

// Forward algorithm
template <typename scalar_t>
__global__ void forward_variables_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> log_probs_a,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> alphas_a,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> labels_a,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> input_lengths_a,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> label_lengths_a,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> cum_a,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> costs_a,
    int blank) {

    const int mb = blockIdx.x * blockDim.x + threadIdx.x;

    if (mb >= log_probs_a.size(0)) {
        return;
    }

    int T = input_lengths_a[mb];     
    int U = label_lengths_a[mb] + 1; 
    int label_offset = cum_a[mb];

    alphas_a[mb][0][0] = 0;
    for (int t = 1; t < T; t++) {
        alphas_a[mb][t][0] = alphas_a[mb][t - 1][0] + log_probs_a[mb][t - 1][0][blank];
    }

    for (int u = 1; u < U; u++) {
        alphas_a[mb][0][u] = alphas_a[mb][0][u - 1] + log_probs_a[mb][0][u - 1][labels_a[label_offset + u - 1]];
    }

    for (int t = 1; t < T; t++) {
        for (int u = 1; u < U; u++) {
            float no_emit = alphas_a[mb][t - 1][u] + log_probs_a[mb][t - 1][u][blank];
            float emit = alphas_a[mb][t][u - 1] + log_probs_a[mb][t][u - 1][labels_a[label_offset + u - 1]];
            alphas_a[mb][t][u] = log_sum_exp(emit, no_emit);
        }
    }
    float forward_ll = alphas_a[mb][T - 1][U - 1] + log_probs_a[mb][T - 1][U - 1][blank];
    costs_a[mb] = -forward_ll;
}

// Backward algorithm
template <typename scalar_t>
__global__ void backward_variables_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> log_probs_a,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> betas_a,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> labels_a,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> input_lengths_a,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> label_lengths_a,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> cum_a,
    int blank) {

    const int mb = blockIdx.x * blockDim.x + threadIdx.x;

    if (mb >= log_probs_a.size(0)) {
        return;
    }

    int T = input_lengths_a[mb];     // Length of utterance (time)
    int U = label_lengths_a[mb] + 1; // Length of transcription
    int label_offset = cum_a[mb];

    betas_a[mb][T - 1][U - 1] = log_probs_a[mb][T - 1][U - 1][blank];
    for (int t = T - 2; t >= 0; t--) {
        betas_a[mb][t][U - 1] = betas_a[mb][t + 1][U - 1] + log_probs_a[mb][t][U - 1][blank];
    }
    for (int u = U - 2; u >= 0; u--) {
        betas_a[mb][T - 1][u] = betas_a[mb][T - 1][u + 1] + log_probs_a[mb][T - 1][u][labels_a[label_offset + u]];
    }

    for (int t = T - 2; t >= 0; t--) {
        for (int u = U - 2; u >= 0; u--) {
            float no_emit = betas_a[mb][t + 1][u] + log_probs_a[mb][t][u][blank];
            float emit = betas_a[mb][t][u + 1] + log_probs_a[mb][t][u][labels_a[label_offset + u]];
            betas_a[mb][t][u] = log_sum_exp(emit, no_emit);
        }
    }
}

template <typename scalar_t>
__global__ void clip_each_value(
	torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> log_probs_a,
	const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> input_lengths_a,
	const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> label_lengths_a,
	float min_value) {
    const int mb = blockIdx.z * blockDim.z + threadIdx.z;
    const int t = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int vocab_size = log_probs_a.size(3);
    const int u = k / vocab_size;
    const int v = k % vocab_size;

    if (mb < log_probs_a.size(0) && t < log_probs_a.size(1) &&
	u < log_probs_a.size(2) && v < log_probs_a.size(3)) {
	int T = input_lengths_a[mb];
        int U = label_lengths_a[mb] + 1;
	float dl_ds = 0;
   // TODO: simplify if-s here to speed up computation
	if (t < T && u < U ) {
        	log_probs_a[mb][t][u][v] = max(log_probs_a[mb][t][u][v], min_value);					
	}
    }

}

void clip_value(
    torch::Tensor log_probs,
    torch::Tensor input_lengths,
    torch::Tensor label_lengths,
    float min_value) {

    const int batch_size = log_probs.size(0);
    const int max_t = log_probs.size(1);
    const int max_u = log_probs.size(2);
    const int vocab_size = log_probs.size(3);

    const dim3 threads3(1024, 1, 1);
    const dim3 blocks3((max_u * vocab_size + threads3.x - 1) / threads3.x,
                       (max_t + threads3.y - 1) / threads3.y,
                       (batch_size + threads3.z - 1) / threads3.z);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(log_probs.scalar_type(), "clip value", ([&] {
        clip_each_value<scalar_t><<<blocks3, threads3>>>(
            log_probs.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
	    input_lengths.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            label_lengths.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            min_value);
    }));

    cudaDeviceSynchronize();
}

void rnnt_cuda(
    torch::Tensor log_probs,  // this is actually tensor containing activations (i.e. logits),
                              // it will be converted to log_probs inside this function
    torch::Tensor labels,         
    torch::Tensor input_lengths,  
    torch::Tensor label_lengths,  
    torch::Tensor costs,          
    torch::Tensor cum,          
    int blank,
    float loss_scale) {

    int batch_size = log_probs.size(0);
    int max_t = log_probs.size(1);
    int max_u = log_probs.size(2);


    // implement log softmax to do the inplace computation even though
    // it is slower than log_softmax in Pytorch, but save memory

    // TODO: can be much faster if run in a single kernel
    //       (actually total slowdown is 2x, because of this)

   // TODO: can we just get max_log_probs?
    // if (sizeof(log_probs[0][0][0][0]) == 4)
#if 0
    torch::Tensor max_log_probs, argmax;
    std::tie(max_log_probs, argmax) = log_probs.max(/*dim=*/3, /*keepdim=*/true);
    log_probs -= max_log_probs;
    log_probs.exp_();
    auto logSumexp = log_probs.sum(/*dim=*/3, /*keepdim=*/true);
    logSumexp.log_();
    clip_value(log_probs,input_lengths, label_lengths, 1e-30);
    log_probs.log_();
    log_probs -= logSumexp;

#endif
    torch::Tensor alphas = torch::empty({batch_size, max_t, max_u}, torch::kCUDA);
    torch::Tensor betas = torch::empty({batch_size, max_t, max_u}, torch::kCUDA);
    
    

    // TODO: maybe add a check for this
    // assuming batch_size is always smaller than number of threads!
    const int threads = batch_size;
    const int blocks = 1;

    // TODO: figure out why not 2 times faster

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(log_probs.scalar_type(), "rnnt_cuda_forward", ([&] {
        forward_variables_kernel<scalar_t><<<blocks, threads>>>(
            log_probs.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            alphas.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            labels.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            input_lengths.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            label_lengths.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            cum.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            costs.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            blank);
    }));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(log_probs.scalar_type(), "rnnt_cuda_backward", ([&] {
        backward_variables_kernel<scalar_t><<<blocks, threads>>>(
            log_probs.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            betas.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            labels.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            input_lengths.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            label_lengths.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            cum.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            blank);
    }));

    cudaDeviceSynchronize();

    compute_grads_cuda(log_probs, alphas, betas, labels, input_lengths,
                       label_lengths, costs, cum, blank, loss_scale);
}
