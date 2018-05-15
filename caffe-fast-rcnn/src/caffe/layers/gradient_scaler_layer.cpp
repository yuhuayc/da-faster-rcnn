#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/gradient_scaler_layer.hpp"

namespace caffe {

template <typename Dtype>
void GradientScalerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape((bottom[0]->shape()));
  coeff_ = this->layer_param_.gradient_scaler_param().scale_factor();
}

template <typename Dtype>
void GradientScalerLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    top[0]->Reshape((bottom[0]->shape()));
}

template <typename Dtype>
void GradientScalerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ShareData(*bottom[0]);
}

template <typename Dtype>
void GradientScalerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const int count = bottom[0]->count();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    caffe_cpu_scale(count, Dtype(coeff_), top_diff, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(GradientScalerLayer);
#endif

INSTANTIATE_CLASS(GradientScalerLayer);
REGISTER_LAYER_CLASS(GradientScaler);

}  // namespace caffe
