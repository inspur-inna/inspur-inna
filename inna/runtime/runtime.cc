#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstdio>
#include <algorithm>
#include <iostream>
#include <iomanip>

#include "runtime.h"


#define MAP_SIZE (32*1024UL)
#define MAP_MASK (MAP_SIZE - 1)
/*
 * man 2 write:
 * On Linux, write() (and similar system calls) will transfer at most
 * 	0x7ffff000 (2,147,479,552) bytes, returning the number of bytes
 *	actually transferred.  (This is true on both 32-bit and 64-bit
 *	systems.)
 */
#define RW_MAX_SIZE	0x7ffff000


struct dname_t {
  const char *ctrl;
  const char *write;
  const char *read;
  const char *wait;
};

const struct dname_t kDevName = {
  "/dev/xdma0_user",
  "/dev/xdma0_h2c_0",
  "/dev/xdma0_c2h_0",
  "/dev/xdma0_events_0"
};


//#ifdef INNA_RUNTIME_DEBUG
template <typename T>
bool CheckEqual(const char* tname, py::array_t<T> test, py::array_t<T> real) {
  auto dtest = test.template unchecked<1>();
  auto dreal = real.template unchecked<1>();
  if (dtest.size() != dreal.size()) {
    throw std::runtime_error("Array shape must equal");
  }

  bool flag = true;
  ssize_t wrong_index;
  for (ssize_t i = 0; i < dtest.size(); ++i) {
    if (dtest(i) != dreal(i)) {
      flag = false;
      wrong_index = i;
      break;
    }
  }

  if (flag) {
    std::cout << tname << " : test passed!" << std::endl;
  } else {
    std::cout << tname << " : test failed at index " << wrong_index << std::endl;
    std::cout << tname << " : data from index " << wrong_index 
              << " show as below :" << std::endl;
    ssize_t end_index = std::min(wrong_index + 32, test.size());
    std::cout << std::hex << std::showbase;
    for (ssize_t i = wrong_index; i < end_index; ++i) {
      std::cout << +dreal(i) << ' ';
    }
    std::cout << std::endl;
    for (ssize_t i = wrong_index; i < end_index; ++i) {
      std::cout << +dtest(i) << ' ';
    }
    std::cout << std::endl;
    std::cout << std::resetiosflags(std::ios_base::basefield);
  }

  return flag;
}
//#endif // end of INNA_RUNTIME_DEBUG


static ssize_t RegisterRW(const char *fname, int fd, 
    uint64_t addr, void *buf, size_t count, char mode) {

  if ((addr % 4) != 0) {
    fprintf(stderr, "addr is not 4x\n");
    perror("check param failed");
    return -EINVAL;
  }

  if ((count % 4) != 0) {
    fprintf(stderr, "count is not 4x\n");
    perror("check param failed");
    return -EINVAL;
  } 
  if (mode != 'w' && mode != 'r') {
    fprintf(stderr, "mode is not w/r\n");
    perror("check param failed");
    return -EINVAL;
  }

  void *map_base = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (map_base == (void *)-1) {
    fprintf(stderr, "%s, mmap failed.\n", fname);
    perror("mmap failed");
    return -EIO;
  }

  void *virt_addr = (uint8_t *)map_base + addr;
  for (size_t i = 0; i < count/4; ++i) {
    if (mode == 'w') {
      *((uint32_t *)virt_addr + i) = *((uint32_t *)buf + i);
    } else {
      *((uint32_t *)buf + i) = *((uint32_t *)virt_addr + i);
    }
  }

  if (munmap(map_base, MAP_SIZE) == -1) {
    fprintf(stderr, "%s, munmap failed.\n", fname);
    perror("munmap failed");
    return -EIO;
  }

  return (ssize_t)count;
}

static ssize_t ReadToBuffer(const char *fname, int fd, 
    uint64_t addr, void *buffer, uint64_t size) {

  ssize_t rc;
  uint64_t count = 0;
  uint8_t *buf = (uint8_t *)buffer;
  off_t offset = addr;

  while (count < size) {
    uint64_t bytes = size - count;

    if (bytes > RW_MAX_SIZE)
      bytes = RW_MAX_SIZE;

    rc = lseek(fd, offset, SEEK_SET);
    if (rc != offset) {
      fprintf(stderr, "%s, seek off 0x%lx != 0x%lx.\n",
          fname, rc, offset);
      perror("seek file");
      return -EIO;
    }

    /* read data from file into memory buffer */
    rc = read(fd, buf, bytes);
    if (rc != bytes) {
      fprintf(stderr, "%s, R off 0x%lx, 0x%lx != 0x%lx.\n",
          fname, count, rc, bytes);
      perror("read file");
      return -EIO;
    }

    count += bytes;
    buf += bytes;
    offset += bytes;
  }	 

  if (count != size) {
    fprintf(stderr, "%s, R failed 0x%lx != 0x%lx.\n",
        fname, count, size);
    return -EIO;
  }
  return count;
}

static ssize_t WriteFromBuffer(const char *fname, int fd, 
    uint64_t addr, void *buffer, uint64_t size) {
  ssize_t rc;
  uint64_t count = 0;
  uint8_t *buf = (uint8_t *)buffer;
  off_t offset = addr;

  while (count < size) {
    uint64_t bytes = size - count;

    if (bytes > RW_MAX_SIZE)
      bytes = RW_MAX_SIZE;

    rc = lseek(fd, offset, SEEK_SET);
    if (rc != offset) {
      fprintf(stderr, "%s, seek off 0x%lx != 0x%lx.\n",
          fname, rc, offset);
      perror("seek file");
      return -EIO;
    }

    /* write data to file from memory buffer */
    rc = write(fd, buf, bytes);
    if (rc != bytes) {
      fprintf(stderr, "%s, W off 0x%lx, 0x%lx != 0x%lx.\n",
          fname, offset, rc, bytes);
      perror("write file");
      return -EIO;
    }

    count += bytes;
    buf += bytes;
    offset += bytes;
  }	 

  if (count != size) {
    fprintf(stderr, "%s, R failed 0x%lx != 0x%lx.\n",
        fname, count, size);
    return -EIO;
  }
  return count;
}


INNARuntime::INNARuntime() {

  fd_ctrl = open(kDevName.ctrl, O_RDWR | O_SYNC);
  if (fd_ctrl == -1) {
    fprintf(stderr, "%s, open failed!\n", kDevName.ctrl);
    exit(1);
  }

  fd_write = open(kDevName.write, O_RDWR);
  if (fd_write == -1) {
    fprintf(stderr, "%s, open failed!\n", kDevName.write);
    exit(1);
  }
  lseek(fd_write, 0UL, SEEK_SET);

  fd_read = open(kDevName.read, O_RDWR);
  if (fd_read == -1) {
    fprintf(stderr, "%s, open failed!\n", kDevName.read);
    exit(1);
  }
  lseek(fd_write, 0UL, SEEK_SET);

  fd_wait = open(kDevName.wait, O_RDWR);
  if (fd_wait == -1) {
    fprintf(stderr, "%s, open failed!\n", kDevName.wait);
    exit(1);
  }
}

INNARuntime::~INNARuntime() {
  close(fd_ctrl);
  close(fd_write);
  close(fd_read);
  close(fd_wait);
}

template <typename T>
ssize_t INNARuntime::WriteRegister(uint64_t addr, py::array_t<T>& buf) {
  if (buf.ndim() != 1) {
    throw std::runtime_error("Number of dimensions must be 1");
  }
  return RegisterRW(kDevName.ctrl, fd_ctrl, addr, (void *)buf.data(), buf.nbytes(), 'w');
}

template <typename T>
py::tuple INNARuntime::ReadRegister(uint64_t addr, uint64_t size) {
  py::array_t<T> buf(size);
  ssize_t count = RegisterRW(kDevName.ctrl, fd_ctrl, addr, (void *)buf.data(), buf.nbytes(), 'r');
  return py::make_tuple(buf, count);
}

template <typename T>
ssize_t INNARuntime::WriteDMA(uint64_t addr, py::array_t<T>& buf) {
  if (buf.ndim() != 1) {
    throw std::runtime_error("Number of dimensions must be 1");
  }
  return WriteFromBuffer(kDevName.write, fd_write, addr, (void *)buf.data(), buf.nbytes());
}

template <typename T>
py::tuple INNARuntime::ReadDMA(uint64_t addr, uint64_t size) {
  py::array_t<T> buf(size);
  ssize_t count = ReadToBuffer(kDevName.read, fd_read, addr, (void *)buf.data(), buf.nbytes());
  return py::make_tuple(buf, count);
}

int INNARuntime::WaitPcieInterupt() {

  int interupt_number;
  ssize_t num = read(fd_wait, (void *)(&interupt_number), 4);
  if (num != 4) {
    fprintf(stderr, "%s, read bytes num not %ld equal 4.\n", kDevName.wait, num);
    perror("read file");
    return -EIO;
  }

  return interupt_number;
}


template <typename T>
void INNARuntime::ReloadNNModel(
    const uint64_t instrs_addr,
    py::array_t<T>& instrs,
    const uint64_t weights_addr,
    py::array_t<T>& weights,
    const uint64_t quant_addr,
    py::array_t<T>& quant_table) {

  WriteDMA(instrs_addr, instrs);
#ifdef INNA_RUNTIME_DEBUG
  py::tuple r_instrs = ReadDMA<T>(instrs_addr, instrs.size());
  CheckEqual("Instruction", (py::array_t<T>)r_instrs[0], instrs);
#endif

  WriteDMA(weights_addr, weights);
#ifdef INNA_RUNTIME_DEBUG
  py::tuple r_weights = ReadDMA<T>(weights_addr, weights.size());
  CheckEqual("Weights", (py::array_t<T>)r_weights[0], weights);
#endif

  WriteDMA(quant_addr, quant_table);
#ifdef INNA_RUNTIME_DEBUG
  py::tuple r_quant = ReadDMA<T>(quant_addr, quant_table.size());
  CheckEqual("Quant table", (py::array_t<T>)r_quant[0], quant_table);
#endif
}

void INNARuntime::SetExtendHWBatch(uint32_t extend_hw_batch) {
  uint64_t r0_addr = 0x50;
  py::array_t<uint32_t> buf(2);
  buf.mutable_at(0) = extend_hw_batch;
  buf.mutable_at(1) = 0;
  WriteRegister(r0_addr, buf);
#ifdef INNA_RUNTIME_DEBUG
  py::tuple r_buf = ReadRegister<uint32_t>(r0_addr, 2);
  CheckEqual("Register R0", (py::array_t<uint32_t>)r_buf[0], buf);
#endif
}

template <typename T>
void INNARuntime::SetInputFeatures(uint64_t ifeature_addr, py::array_t<T>& ifeatures) {
  WriteDMA(ifeature_addr, ifeatures);
#ifdef INNA_RUNTIME_DEBUG
  py::tuple r_features = ReadDMA<T>(ifeature_addr, ifeatures.size());
  CheckEqual("Input feature", (py::array_t<T>)r_features[0], ifeatures);
#endif
}

void INNARuntime::Run() {
  uint64_t ctrl_addr = 0x0;
  py::array_t<uint32_t> buf(1);

  // reset device
#ifdef INNA_RUNTIME_DEBUG
  std::cout << "Run : reset device" << std::endl;
#endif
  uint32_t reset_cmd1 = ((0x3 & 0xFF) << 24) | 0x1;
  uint32_t reset_cmd0 = ((0x3 & 0xFF) << 24) | 0x0;
  buf.mutable_at(0) = reset_cmd1;
  WriteRegister(ctrl_addr, buf);
  buf.mutable_at(0) = reset_cmd0;
  WriteRegister(ctrl_addr, buf);

  // start
#ifdef INNA_RUNTIME_DEBUG
  std::cout << "Run : task start" << std::endl;
#endif
  uint32_t start_cmd = ((0x0 & 0xFF) << 24);
  buf.mutable_at(0) = start_cmd;
  WriteRegister(ctrl_addr, buf);
}

int INNARuntime::Wait() {
  return WaitPcieInterupt();
}

template <typename T>
py::array_t<T> INNARuntime::GetOutputFeatures(uint64_t ofeature_addr, uint64_t size) {
  py::tuple r_features = ReadDMA<T>(ofeature_addr, size);
  return (py::array_t<T>)r_features[0];
}


PYBIND11_MODULE(_runtime, m) {
  m.def("CheckEqual", &CheckEqual<uint8_t>);
  m.def("CheckEqual", &CheckEqual<uint32_t>);
  m.def("CheckEqual", &CheckEqual<float>);
  m.def("CheckEqual", &CheckEqual<double>);

  py::class_<INNARuntime> runtime(m, "INNARuntime");
  runtime
    .def(py::init<>())

    .def("WriteRegister", &INNARuntime::WriteRegister<uint8_t>, py::arg("addr"), py::arg("buf"))
    .def("WriteRegister", &INNARuntime::WriteRegister<uint16_t>, py::arg("addr"), py::arg("buf"))
    .def("WriteRegister", &INNARuntime::WriteRegister<uint32_t>, py::arg("addr"), py::arg("buf"))
    .def("WriteRegister", &INNARuntime::WriteRegister<uint64_t>, py::arg("addr"), py::arg("buf"))
    .def("ReadRegisterU8", &INNARuntime::ReadRegister<uint8_t>,
        py::arg("addr"), py::arg("size"), py::return_value_policy::reference)
    .def("ReadRegisterU16", &INNARuntime::ReadRegister<uint16_t>,
        py::arg("addr"), py::arg("size"), py::return_value_policy::reference)
    .def("ReadRegisterU32", &INNARuntime::ReadRegister<uint32_t>,
        py::arg("addr"), py::arg("size"), py::return_value_policy::reference)
    .def("ReadRegisterU64", &INNARuntime::ReadRegister<uint64_t>,
        py::arg("addr"), py::arg("size"), py::return_value_policy::reference)
    .def("WriteDMA", &INNARuntime::WriteDMA<uint8_t>, py::arg("addr"), py::arg("buf"))
    .def("ReadDMA", &INNARuntime::ReadDMA<uint8_t>,
        py::arg("addr"), py::arg("size"), py::return_value_policy::reference)
    .def("WaitPcieInterupt", &INNARuntime::WaitPcieInterupt)

    .def("ReloadNNModel", &INNARuntime::ReloadNNModel<uint8_t>, 
        py::arg("instrs_addr"), py::arg("instrs"),
        py::arg("weights_addr"), py::arg("weights"),
        py::arg("quant_addr"), py::arg("quant_table"))
    .def("SetExtendHWBatch", &INNARuntime::SetExtendHWBatch, py::arg("extend_hw_batch"))
    .def("SetInputFeatures", &INNARuntime::SetInputFeatures<uint8_t>, py::arg("ifeature_addr"), py::arg("ifeatures"))
    .def("Run", &INNARuntime::Run)
    .def("Wait", &INNARuntime::Wait)
    .def("GetOutputFeatures", &INNARuntime::GetOutputFeatures<uint8_t>, 
        py::arg("ofeature_addr"), py::arg("size"), py::return_value_policy::reference);
}

