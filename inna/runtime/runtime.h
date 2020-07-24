#ifndef RUNTIME_H_
#define RUNTIME_H_

#include <string>
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#define INNA_RUNTIME_DEBUG

namespace py = pybind11;


/*! \class INNARuntime
 *  \brief A Runtime Class
 *
 *  A class supply device management interface
 */
class INNARuntime {
public:
  INNARuntime();

  ~INNARuntime();

  template <typename T>
  ssize_t WriteRegister(uint64_t addr, py::array_t<T>& buf);

  template <typename T>
  py::tuple ReadRegister(uint64_t addr, uint64_t size);

  template <typename T>
  ssize_t WriteDMA(uint64_t addr, py::array_t<T>& buf);

  template <typename T>
  py::tuple ReadDMA(uint64_t addr, uint64_t size);

  int WaitPcieInterupt();

  template <typename T>
  void ReloadNNModel(
      const uint64_t instrs_addr,
      py::array_t<T>& instrs,
      const uint64_t weights_addr,
      py::array_t<T>& weights,
      const uint64_t quant_addr,
      py::array_t<T>& quant_table);

  void SetExtendHWBatch(uint32_t extend_hw_batch);

  template <typename T>
  void SetInputFeatures(uint64_t ifeature_addr, py::array_t<T>& ifeatures);

  void Run();

  int Wait();

  template <typename T>
  py::array_t<T> GetOutputFeatures(uint64_t ofeature_addr, uint64_t size);

private:
  int fd_ctrl;
  int fd_write;
  int fd_read;
  int fd_wait;
};



#endif // end of RUNTIME_H_
