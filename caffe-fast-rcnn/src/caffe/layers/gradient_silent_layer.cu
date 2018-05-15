#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/gradient_silent_layer.hpp"

namespace caffe {

template <typename Dtype>
void GradientSilentLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ShareData(*bottom[0]);
}

template <typename Dtype>
void GradientSilentLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const int count = top[0]->count();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    const Dtype* need_backprop_handle = bottom[1]->cpu_data();
    const Dtype need_backprop = need_backprop_handle[0];

    if(need_backprop!=Dtype(0)){
        caffe_gpu_scale(count, Dtype(1), top_diff, bottom_diff);
        }
    else{
        caffe_gpu_scale(count, Dtype(0), top_diff, bottom_diff);
        }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(GradientSilentLayer);

}  // namespace caffe
