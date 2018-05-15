#ifndef CAFFE_GRADIENT_SCALER_LAYER_HPP_
#define CAFFE_GRADIENT_SCALER_LAYER_HPP_


#include <vector>


#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"

#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class GradientScalerLayer : public Layer<Dtype> {
 public:
  explicit GradientScalerLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "GradientScaler"; }

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  float coeff_;
};
}

#endif  // CAFFE_GRADIENT_SCALER_LAYER_HPP_
