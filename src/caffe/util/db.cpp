
#include "caffe/util/db.hpp"

#include <sys/stat.h>

#include <string>

namespace caffe { namespace db {


void Cursor::SeekToFirst() {
    if (in_ && in_->is_open()) {
      in_->close();
    }
    //LOG(INFO) << "reset ifstream " << path_;
    in_ = new std::ifstream(path_.c_str(),
            std::ifstream::in|std::ifstream::binary);
    Next();
  }

void Cursor::Next() {
  valid_ = false;
  CHECK(in_->is_open()) << "file is not open!" << path_;

  uint32_t record_size = 0, key_size = 0, value_size = 0;
  in_->read(reinterpret_cast<char*>(&record_size), sizeof record_size);
  if (in_->gcount() != (sizeof record_size) || record_size > MAX_BUF) {
    CHECK(in_->eof() && record_size <= MAX_BUF)
      <<"record_size read error: gcount\t"
      << in_->gcount() << "\trecord_size\t" << record_size;
    return;
  }

  in_->read(reinterpret_cast<char*>(&key_size), sizeof key_size);
  CHECK(in_->gcount() == sizeof key_size && key_size <= MAX_BUF)
    << "key_size read error: gcount\t"
    << in_->gcount() << "\tkey_size\t" << key_size;

  key_.resize(key_size);
  in_->read(&key_[0], key_size);
  CHECK(in_->gcount() == key_size)
    << "key read error: gcount\t"
    << in_->gcount() << "\tkey_size\t" << key_size;

  in_->read(reinterpret_cast<char*>(&value_size), sizeof value_size);
  CHECK(in_->gcount() == sizeof value_size && value_size <= MAX_BUF)
    << "value_size read error: gcount\t"
    << in_->gcount() << "\tvalue_size\t" << value_size;

  value_.resize(value_size);
  in_->read(&value_[0], value_size);
  CHECK(in_->gcount() == value_size)
    << "value read error: gcount\t"
    << in_->gcount() << "\tvalue_size\t" << value_size;

  valid_ = true;
}

void Transaction::Put(const string& key, const string& value) {
  try {
    uint32_t key_size = key.size(), value_size = value.size();
    uint32_t record_size = key_size + value_size
        + sizeof key_size + sizeof value_size;
    out_->write(reinterpret_cast<char*>(&record_size), sizeof record_size);
    out_->write(reinterpret_cast<char*>(&key_size), sizeof key_size);
    out_->write(key.data(), key_size);
    out_->write(reinterpret_cast<char*>(&value_size), sizeof value_size);
    out_->write(value.data(), value_size);
  } catch(std::ios_base::failure& e) {
    LOG(FATAL) << "Exception: "
        << e.what() << " rdstate: " << out_->rdstate() << '\n';
  }
}

Transaction* DB::NewTransaction() {
  if (!this->out_) {
    out_ = new std::ofstream();
    out_->open(this->path_.c_str(),
            std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
    out_->exceptions(out_->exceptions() | std::ios::failbit);
    LOG(INFO) << "Output created: " << path_ << std::endl;
  }
  return new Transaction(this->out_);
}
}  // namespace db
}  // namespace caffe

