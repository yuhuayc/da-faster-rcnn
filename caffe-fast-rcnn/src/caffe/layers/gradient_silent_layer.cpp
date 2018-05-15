#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/gradient_silent_layer.hpp"


namespace caffe {

template <typename Dtype>
void GradientSilentLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape((bottom[0]->shape()));
}

template <typename Dtype>
void GradientSilentLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    top[0]->Reshape((bottom[0]->shape()));
}

template <typename Dtype>
void GradientSilentLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ShareData(*bottom[0]);
}

template <typename Dtype>
void GradientSilentLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const int count = bottom[0]->count();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    caffe_cpu_scale(count, Dtype(0), top_diff, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(GradientSilentLayer);
#endif

INSTANTIATE_CLASS(GradientSilentLayer);
REGISTER_LAYER_CLASS(GradientSilent);

}  // namespace caffe
