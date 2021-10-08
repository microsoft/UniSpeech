#include <torch/extension.h>

#include <vector>

void rnnt_cuda(
    torch::Tensor acts,
    torch::Tensor labels,         
    torch::Tensor input_lengths,  
    torch::Tensor label_lengths,  
    torch::Tensor costs,         
    torch::Tensor cum,         
    int blank,
    float loss_scale);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(!x.type().is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT_ON_CUDA(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_INPUT_ON_CPU(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)


void transducer(
    torch::Tensor acts,
    torch::Tensor labels,         
    torch::Tensor input_lengths,  
    torch::Tensor label_lengths,  
    torch::Tensor costs,          
    torch::Tensor cum,          
    int blank,
    float loss_scale) {

    CHECK_INPUT_ON_CUDA(acts);
    CHECK_INPUT_ON_CUDA(labels);
    CHECK_INPUT_ON_CUDA(input_lengths);
    CHECK_INPUT_ON_CUDA(label_lengths);
    CHECK_INPUT_ON_CUDA(costs);
    CHECK_INPUT_ON_CUDA(cum);

    rnnt_cuda(acts, labels, input_lengths,
              label_lengths, costs, cum, blank, loss_scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("transducer", &transducer, "RNNT transducer (CUDA)");
}
