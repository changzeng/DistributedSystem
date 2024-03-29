/**
 * Autogenerated by Thrift Compiler (0.12.0)
 *
 * DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
 *  @generated
 */
#include "dist_lr_types.h"

#include <algorithm>
#include <ostream>

#include <thrift/TToString.h>




DeltaW::~DeltaW() throw() {
}


void DeltaW::__set_version(const int32_t val) {
  this->version = val;
}

void DeltaW::__set_parameters(const std::vector<double> & val) {
  this->parameters = val;
}
std::ostream& operator<<(std::ostream& out, const DeltaW& obj)
{
  obj.printTo(out);
  return out;
}


uint32_t DeltaW::read(::apache::thrift::protocol::TProtocol* iprot) {

  ::apache::thrift::protocol::TInputRecursionTracker tracker(*iprot);
  uint32_t xfer = 0;
  std::string fname;
  ::apache::thrift::protocol::TType ftype;
  int16_t fid;

  xfer += iprot->readStructBegin(fname);

  using ::apache::thrift::protocol::TProtocolException;


  while (true)
  {
    xfer += iprot->readFieldBegin(fname, ftype, fid);
    if (ftype == ::apache::thrift::protocol::T_STOP) {
      break;
    }
    switch (fid)
    {
      case -1:
        if (ftype == ::apache::thrift::protocol::T_I32) {
          xfer += iprot->readI32(this->version);
          this->__isset.version = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      case -2:
        if (ftype == ::apache::thrift::protocol::T_LIST) {
          {
            this->parameters.clear();
            uint32_t _size0;
            ::apache::thrift::protocol::TType _etype3;
            xfer += iprot->readListBegin(_etype3, _size0);
            this->parameters.resize(_size0);
            uint32_t _i4;
            for (_i4 = 0; _i4 < _size0; ++_i4)
            {
              xfer += iprot->readDouble(this->parameters[_i4]);
            }
            xfer += iprot->readListEnd();
          }
          this->__isset.parameters = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      default:
        xfer += iprot->skip(ftype);
        break;
    }
    xfer += iprot->readFieldEnd();
  }

  xfer += iprot->readStructEnd();

  return xfer;
}

uint32_t DeltaW::write(::apache::thrift::protocol::TProtocol* oprot) const {
  uint32_t xfer = 0;
  ::apache::thrift::protocol::TOutputRecursionTracker tracker(*oprot);
  xfer += oprot->writeStructBegin("DeltaW");

  xfer += oprot->writeFieldBegin("parameters", ::apache::thrift::protocol::T_LIST, -2);
  {
    xfer += oprot->writeListBegin(::apache::thrift::protocol::T_DOUBLE, static_cast<uint32_t>(this->parameters.size()));
    std::vector<double> ::const_iterator _iter5;
    for (_iter5 = this->parameters.begin(); _iter5 != this->parameters.end(); ++_iter5)
    {
      xfer += oprot->writeDouble((*_iter5));
    }
    xfer += oprot->writeListEnd();
  }
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldBegin("version", ::apache::thrift::protocol::T_I32, -1);
  xfer += oprot->writeI32(this->version);
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldStop();
  xfer += oprot->writeStructEnd();
  return xfer;
}

void swap(DeltaW &a, DeltaW &b) {
  using ::std::swap;
  swap(a.version, b.version);
  swap(a.parameters, b.parameters);
  swap(a.__isset, b.__isset);
}

DeltaW::DeltaW(const DeltaW& other6) {
  version = other6.version;
  parameters = other6.parameters;
  __isset = other6.__isset;
}
DeltaW& DeltaW::operator=(const DeltaW& other7) {
  version = other7.version;
  parameters = other7.parameters;
  __isset = other7.__isset;
  return *this;
}
void DeltaW::printTo(std::ostream& out) const {
  using ::apache::thrift::to_string;
  out << "DeltaW(";
  out << "version=" << to_string(version);
  out << ", " << "parameters=" << to_string(parameters);
  out << ")";
}


DataSet::~DataSet() throw() {
}


void DataSet::__set_sample_list(const std::vector<std::vector<double> > & val) {
  this->sample_list = val;
}

void DataSet::__set_labels(const std::vector<double> & val) {
  this->labels = val;
}
std::ostream& operator<<(std::ostream& out, const DataSet& obj)
{
  obj.printTo(out);
  return out;
}


uint32_t DataSet::read(::apache::thrift::protocol::TProtocol* iprot) {

  ::apache::thrift::protocol::TInputRecursionTracker tracker(*iprot);
  uint32_t xfer = 0;
  std::string fname;
  ::apache::thrift::protocol::TType ftype;
  int16_t fid;

  xfer += iprot->readStructBegin(fname);

  using ::apache::thrift::protocol::TProtocolException;


  while (true)
  {
    xfer += iprot->readFieldBegin(fname, ftype, fid);
    if (ftype == ::apache::thrift::protocol::T_STOP) {
      break;
    }
    switch (fid)
    {
      case -1:
        if (ftype == ::apache::thrift::protocol::T_LIST) {
          {
            this->sample_list.clear();
            uint32_t _size8;
            ::apache::thrift::protocol::TType _etype11;
            xfer += iprot->readListBegin(_etype11, _size8);
            this->sample_list.resize(_size8);
            uint32_t _i12;
            for (_i12 = 0; _i12 < _size8; ++_i12)
            {
              {
                this->sample_list[_i12].clear();
                uint32_t _size13;
                ::apache::thrift::protocol::TType _etype16;
                xfer += iprot->readListBegin(_etype16, _size13);
                this->sample_list[_i12].resize(_size13);
                uint32_t _i17;
                for (_i17 = 0; _i17 < _size13; ++_i17)
                {
                  xfer += iprot->readDouble(this->sample_list[_i12][_i17]);
                }
                xfer += iprot->readListEnd();
              }
            }
            xfer += iprot->readListEnd();
          }
          this->__isset.sample_list = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      case -2:
        if (ftype == ::apache::thrift::protocol::T_LIST) {
          {
            this->labels.clear();
            uint32_t _size18;
            ::apache::thrift::protocol::TType _etype21;
            xfer += iprot->readListBegin(_etype21, _size18);
            this->labels.resize(_size18);
            uint32_t _i22;
            for (_i22 = 0; _i22 < _size18; ++_i22)
            {
              xfer += iprot->readDouble(this->labels[_i22]);
            }
            xfer += iprot->readListEnd();
          }
          this->__isset.labels = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      default:
        xfer += iprot->skip(ftype);
        break;
    }
    xfer += iprot->readFieldEnd();
  }

  xfer += iprot->readStructEnd();

  return xfer;
}

uint32_t DataSet::write(::apache::thrift::protocol::TProtocol* oprot) const {
  uint32_t xfer = 0;
  ::apache::thrift::protocol::TOutputRecursionTracker tracker(*oprot);
  xfer += oprot->writeStructBegin("DataSet");

  xfer += oprot->writeFieldBegin("labels", ::apache::thrift::protocol::T_LIST, -2);
  {
    xfer += oprot->writeListBegin(::apache::thrift::protocol::T_DOUBLE, static_cast<uint32_t>(this->labels.size()));
    std::vector<double> ::const_iterator _iter23;
    for (_iter23 = this->labels.begin(); _iter23 != this->labels.end(); ++_iter23)
    {
      xfer += oprot->writeDouble((*_iter23));
    }
    xfer += oprot->writeListEnd();
  }
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldBegin("sample_list", ::apache::thrift::protocol::T_LIST, -1);
  {
    xfer += oprot->writeListBegin(::apache::thrift::protocol::T_LIST, static_cast<uint32_t>(this->sample_list.size()));
    std::vector<std::vector<double> > ::const_iterator _iter24;
    for (_iter24 = this->sample_list.begin(); _iter24 != this->sample_list.end(); ++_iter24)
    {
      {
        xfer += oprot->writeListBegin(::apache::thrift::protocol::T_DOUBLE, static_cast<uint32_t>((*_iter24).size()));
        std::vector<double> ::const_iterator _iter25;
        for (_iter25 = (*_iter24).begin(); _iter25 != (*_iter24).end(); ++_iter25)
        {
          xfer += oprot->writeDouble((*_iter25));
        }
        xfer += oprot->writeListEnd();
      }
    }
    xfer += oprot->writeListEnd();
  }
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldStop();
  xfer += oprot->writeStructEnd();
  return xfer;
}

void swap(DataSet &a, DataSet &b) {
  using ::std::swap;
  swap(a.sample_list, b.sample_list);
  swap(a.labels, b.labels);
  swap(a.__isset, b.__isset);
}

DataSet::DataSet(const DataSet& other26) {
  sample_list = other26.sample_list;
  labels = other26.labels;
  __isset = other26.__isset;
}
DataSet& DataSet::operator=(const DataSet& other27) {
  sample_list = other27.sample_list;
  labels = other27.labels;
  __isset = other27.__isset;
  return *this;
}
void DataSet::printTo(std::ostream& out) const {
  using ::apache::thrift::to_string;
  out << "DataSet(";
  out << "sample_list=" << to_string(sample_list);
  out << ", " << "labels=" << to_string(labels);
  out << ")";
}


HostInfo::~HostInfo() throw() {
}


void HostInfo::__set_host(const std::string& val) {
  this->host = val;
}

void HostInfo::__set_port(const int32_t val) {
  this->port = val;
}
std::ostream& operator<<(std::ostream& out, const HostInfo& obj)
{
  obj.printTo(out);
  return out;
}


uint32_t HostInfo::read(::apache::thrift::protocol::TProtocol* iprot) {

  ::apache::thrift::protocol::TInputRecursionTracker tracker(*iprot);
  uint32_t xfer = 0;
  std::string fname;
  ::apache::thrift::protocol::TType ftype;
  int16_t fid;

  xfer += iprot->readStructBegin(fname);

  using ::apache::thrift::protocol::TProtocolException;


  while (true)
  {
    xfer += iprot->readFieldBegin(fname, ftype, fid);
    if (ftype == ::apache::thrift::protocol::T_STOP) {
      break;
    }
    switch (fid)
    {
      case -1:
        if (ftype == ::apache::thrift::protocol::T_STRING) {
          xfer += iprot->readString(this->host);
          this->__isset.host = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      case -2:
        if (ftype == ::apache::thrift::protocol::T_I32) {
          xfer += iprot->readI32(this->port);
          this->__isset.port = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      default:
        xfer += iprot->skip(ftype);
        break;
    }
    xfer += iprot->readFieldEnd();
  }

  xfer += iprot->readStructEnd();

  return xfer;
}

uint32_t HostInfo::write(::apache::thrift::protocol::TProtocol* oprot) const {
  uint32_t xfer = 0;
  ::apache::thrift::protocol::TOutputRecursionTracker tracker(*oprot);
  xfer += oprot->writeStructBegin("HostInfo");

  xfer += oprot->writeFieldBegin("port", ::apache::thrift::protocol::T_I32, -2);
  xfer += oprot->writeI32(this->port);
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldBegin("host", ::apache::thrift::protocol::T_STRING, -1);
  xfer += oprot->writeString(this->host);
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldStop();
  xfer += oprot->writeStructEnd();
  return xfer;
}

void swap(HostInfo &a, HostInfo &b) {
  using ::std::swap;
  swap(a.host, b.host);
  swap(a.port, b.port);
  swap(a.__isset, b.__isset);
}

HostInfo::HostInfo(const HostInfo& other28) {
  host = other28.host;
  port = other28.port;
  __isset = other28.__isset;
}
HostInfo& HostInfo::operator=(const HostInfo& other29) {
  host = other29.host;
  port = other29.port;
  __isset = other29.__isset;
  return *this;
}
void HostInfo::printTo(std::ostream& out) const {
  using ::apache::thrift::to_string;
  out << "HostInfo(";
  out << "host=" << to_string(host);
  out << ", " << "port=" << to_string(port);
  out << ")";
}


