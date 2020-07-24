#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <cstdio>
#include <stdint.h>
#include <cstring>
#include <cerrno>


/*
 * man 2 write:
 * On Linux, write() (and similar system calls) will transfer at most
 * 	0x7ffff000 (2,147,479,552) bytes, returning the number of bytes
 *	actually transferred.  (This is true on both 32-bit and 64-bit
 *	systems.)
 */
#define RW_MAX_SIZE	0x7ffff000


static ssize_t write_from_buffer(const char *fname, int fd, 
    uint64_t addr, uint8_t *buffer, uint64_t size) {
  ssize_t rc;
  uint64_t count = 0;
  uint8_t *buf = buffer;
  off_t offset = addr;

  while (count < size) {
    uint64_t bytes = size - count;

    if (bytes > RW_MAX_SIZE)
      bytes = RW_MAX_SIZE;

    if (offset) {
      rc = lseek(fd, offset, SEEK_SET);
      if (rc != offset) {
        fprintf(stderr, "%s, seek off 0x%lx != 0x%lx.\n",
            fname, rc, offset);
        perror("seek file");
        return -EIO;
      }
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


int main(int argc, char *argv[]) {

  if (argc != 4) {
    fprintf(stderr, "usage : %s <outfile> <size(G/M/K/B)> <start>\n", argv[0]);
    exit(1);
  }

  char *ofname = argv[1];
  size_t size;
  size_t len = strlen(argv[2]);
  switch (argv[2][len - 1]) {
    case 'G':
      argv[2][len - 1] = '\0';
      size = strtoul(argv[2], 0, 0) * 1024 * 1024 * 1024;
      break;
    case 'M':
      argv[2][len - 1] = '\0';
      size = strtoul(argv[2], 0, 0) * 1024 * 1024;
      break;
    case 'K':
      argv[2][len - 1] = '\0';
      size = strtoul(argv[2], 0, 0) * 1024;
      break;
    case 'B':
      argv[2][len - 1] = '\0';
      size = strtoul(argv[2], 0, 0);
      break;
    default:
      size = strtoul(argv[2], 0, 0);
      break;
  }
  size = (uint64_t)((size + 7) / 8) * 8;
  uint64_t start = strtoul(argv[3], 0, 0);

  if (access(ofname, F_OK) == 0) remove(ofname);
  int fd = open(ofname, O_WRONLY | O_CREAT, 0644);
  if (fd == -1) {
    printf("Error: open failed!\n");
    exit(1);
  }

  lseek(fd, 0L, SEEK_SET);
  uint64_t buf_size = (uint64_t)(size / 8);
  uint64_t *buf = new uint64_t[buf_size];
  for (size_t i = 0; i < buf_size; ++i) buf[i] = start + i;

  ssize_t rc = write_from_buffer(ofname, fd, 0, (uint8_t *)buf, size);
  if (rc < 0) perror("write file");

  delete[] buf;
  close(fd);

  return 0;
}
