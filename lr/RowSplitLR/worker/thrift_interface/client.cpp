//
// Created by changzeng on 2019/9/8.
//

#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <thrift/stdcxx.h>
#include <thrift/transport/TSocket.h>
#include <thrift/transport/TTransportUtils.h>
#include <thrift/protocol/TBinaryProtocol.h>

#include "../thrift_interface/ParameterServer.h"
#include "../thrift_interface/dist_lr_types.h"

using std::cout;
using std::endl;
using std::vector;
using std::string;
using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;

typedef ::apache::thrift::stdcxx::shared_ptr<TTransport> transport_ptr;

class DistLRWorker{
private:
    HostInfo host_info;
    DataSet sample_list;
    DeltaW parameters;
    int sleep_seconds=10;
public:
    explicit DistLRWorker(const string& host="localhost", int port=9090){
        host_info.host=host;
        host_info.port=port;
    }

    ParameterServerClient new_connection(transport_ptr& ptr){
        ::apache::thrift::stdcxx::shared_ptr<TTransport> socket(new TSocket("localhost", 9090));
        ::apache::thrift::stdcxx::shared_ptr<TTransport> transport(new TBufferedTransport(socket));
        ::apache::thrift::stdcxx::shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
        transport->open();
        return ParameterServerClient(protocol);
    }

    void wait(){
        // connect to server
        transport_ptr transport;
        ParameterServerClient client = new_connection(transport);
        while(!client.connect_to_master(host_info)){
            cout<<"connect to master error, sleep "<<sleep_seconds<<" seconds"<<endl;
            sleep(10);
        }

        // do some work
        try{
            while(true){
                client.pull_dataset(sample_list);
                if ((int)sample_list.sample_list.size()==0 || (int)sample_list.labels.size()==0){
                    cout<<"Fetch nothing sleeping..."<<endl;
                    sleep(sleep_seconds);
                }else{
                    do_work(client);
                }
            }
        }catch(TException& tx){
            cout<<"Error: "<<tx.what()<<endl;
        }
        transport->close();
    }

    void clear_dataset(){
        sample_list.sample_list.clear();
        sample_list.labels.clear();
    }

    static double sigmoid(double x){
        return (double)1.0/(1+exp(-x));
    }

    static double predict_one(const vector<double>& fea, const vector<double>& w){
        double p_val=0;
        for(int fea_ind=0; fea_ind<(int)fea.size()-1; fea_ind++){
            p_val += fea[fea_ind]*w[fea_ind];
        }
        return sigmoid(p_val+w.back());
    }

    void do_work(ParameterServerClient& client){
        DeltaW weights;
        client.pull_parameters(weights);
        DeltaW delta;
        while(true){
            if((int)weights.parameters.size()==0) break;
            delta.parameters.resize(weights.parameters.size());
            int data_size=(int)sample_list.sample_list.size();
            int fea_dim=(int)weights.parameters.size();
            for(int i=0; i<fea_dim; i++) delta.parameters[i]=0;
            for(int i=0; i<data_size; i++){
                double p_val = predict_one(sample_list.sample_list[i], weights.parameters);
                for(int j=0;j<fea_dim-1; j++){
                    delta.parameters[j] -= (sample_list.labels[i]-p_val)*sample_list.sample_list[i][j];
                }
                delta.parameters[fea_dim-1] -= (sample_list.labels[i]-p_val);
            }
            for(int i=0; i<fea_dim; i++) delta.parameters[i]/=(double)data_size;
            client.push_delta(delta);
        }
    }

    void commit_dataset(){
        transport_ptr transport;
        ParameterServerClient client=new_connection(transport);
        DataSet local_dataset;
        read_csv(local_dataset, "imdb.csv");
        client.push_dataset(local_dataset);
        transport->close();
    }

    static void read_csv(DataSet& dataset, const string& file_name="iris.csv"){
        cout<<"starting read csv file..."<<endl;
        string full_file_name = "/home/changzeng/CLionProjects/LogisticRegression/"+file_name;
        std::ifstream reader(full_file_name);
        string line_str;
        int line_num=0, col_num=0, col_ind=0;
        vector<double> tmp_fea;
        while(getline(reader, line_str)){
            std::stringstream ss(line_str);
            string str;
            col_ind=0;
            tmp_fea.clear();
            while(getline(ss, str, ',')){
                if(line_num==0){
                    ++col_num;
                }else{
                    if(0<col_ind && col_ind<col_num){
                        if(col_ind==col_num-1){
                            dataset.labels.push_back(atof(str.c_str()));
                        }else{
                            tmp_fea.push_back(atof(str.c_str()));
                        }
                    }
                }
                ++col_ind;
            }
            if(line_num!=0) dataset.sample_list.push_back(tmp_fea);
            // if(line_num>=50) break;
            ++line_num;
        }
        reader.close();
        cout<<"read csv file done"<<endl;
        cout<<dataset.sample_list[0].size()<<endl;
        cout<<dataset.sample_list.size()<<endl;
    }
};

bool char_array_cmp(const char* char_1, const char* char_2, int len){
    for(int i=0; i<len; i++){
        if(char_1[i]=='\0' && char_2[i]=='\0') return true;
        if(char_1[i]=='\0' || char_2[i]=='\0') return false;
    }
    return false;
}

int main(int argc, char **argv) {
    DistLRWorker worker;
    string run_mode;
    if(argc>=2){
        run_mode = string(argv[1]);
        if(run_mode=="wait") worker.wait();
        if(run_mode=="commit") worker.commit_dataset();
    }
    return 0;
}
