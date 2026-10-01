// Int-array semantic trace runtime. Compiled only into temporary animation builds.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <vector>
namespace sort_trace {
struct Variable { const char *name; long long value; };
using Variables = std::initializer_list<Variable>;
struct Observed { int value, id, position; };
struct Binding { int frame, key; Observed observed; };
struct Saved { int frame,key,left,right,value,id; };
int *base=nullptr, length=0, ids[12], before[12], before_ids[12], frame=0, next_frame=0, sequence=0;
FILE *file=nullptr;
std::vector<Binding> bindings;
std::vector<Saved> saved;
void fail(const char *message) {std::fprintf(stderr,"[trace error] %s\n",message);std::exit(2);}
struct Frame {
 int previous,own;
 Frame():previous(frame),own(++next_frame){frame=own;}
 ~Frame(){bindings.erase(std::remove_if(bindings.begin(),bindings.end(),[&](const Binding &v){return v.frame==own;}),bindings.end());frame=previous;}
};
void list(const int *values){std::fputc('[',file);for(int i=0;i<length;++i)std::fprintf(file,"%s%d",i?",":"",values[i]);std::fputc(']',file);}
void bind(int *a,int n){if(n<1||n>12)fail("array size");base=a;length=n;for(int i=0;i<n;++i)ids[i]=i;file=std::fopen("sort-trace.jsonl","w");if(!file)fail("trace file");std::fprintf(file,"{\"kind\":\"header\",\"version\":1,\"initial\":");list(base);std::fputs("}\n",file);}
int slot(int &value){for(int i=0;i<length;++i)if(base+i==&value)return i;fail("array access outside registered range");return -1;}
int boundary(int *pointer){for(int i=0;i<=length;++i)if(base+i==pointer)return i;fail("rotate boundary outside array");return -1;}
void capture(){for(int i=0;i<length;++i){before[i]=base[i];before_ids[i]=ids[i];}}
Observed element(int &value){int p=slot(value);return {value,ids[p],p};}
void remember(int key,Observed v){if(!key)return;for(auto &b:bindings)if(b.frame==frame&&b.key==key){b.observed=v;return;}bindings.push_back({frame,key,v});}
Observed scalar(int value,int key){for(const auto &b:bindings)if(b.frame==frame&&b.key==key&&b.observed.value==value){Observed v=b.observed;v.position=-1;for(int i=0;i<length;++i)if(ids[i]==v.id&&base[i]==value){v.position=i;break;}return v;}return {value,-1,-1};}
void emit(const char *kind,int line,int column,Variables vars,int left=-1,int right=-1,int middle=-1,int x=0,int y=0,const char *op="",int result=-1,int bucket=-1,int count=0){
 if(++sequence>10000)fail("trace event limit 10000");
 std::fprintf(file,"{\"kind\":\"%s\",\"seq\":%d,\"line\":%d,\"column\":%d,\"left\":%d,\"right\":%d,\"middle\":%d,\"x\":%d,\"y\":%d,\"operator\":\"%s\",\"result\":%d,\"bucket\":%d,\"count\":%d,\"before\":",kind,sequence,line,column,left,right,middle,x,y,op,result,bucket,count);
 list(before);std::fputs(",\"values\":",file);list(base);std::fputs(",\"beforeIds\":",file);list(before_ids);std::fputs(",\"ids\":",file);list(ids);
 std::fputs(",\"variables\":{",file);bool first=true;for(const auto &v:vars){std::fprintf(file,"%s\"%s\":%lld",first?"":",",v.name,v.value);first=false;}std::fputs("}}\n",file);
 if(std::ftell(file)>8*1024*1024)fail("trace byte limit 8 MiB");
}
bool compare(Observed left,Observed right,const char *op,int line,int column,Variables vars){bool r=false;if(!std::strcmp(op,">"))r=left.value>right.value;else if(!std::strcmp(op,"<"))r=left.value<right.value;else if(!std::strcmp(op,">="))r=left.value>=right.value;else if(!std::strcmp(op,"<="))r=left.value<=right.value;else if(!std::strcmp(op,"=="))r=left.value==right.value;else if(!std::strcmp(op,"!="))r=left.value!=right.value;else fail("comparison operator");capture();emit("compare",line,column,vars,left.position,right.position,-1,left.value,right.value,op,r);return r;}
void exchange(int &left,int &right,int line,int column,Variables vars){int l=slot(left),r=slot(right),x=left,y=right;capture();std::swap(left,right);std::swap(ids[l],ids[r]);emit("swap",line,column,vars,l,r,-1,x,y);}
int read(int &value,int key,int line,int column,Variables vars){Observed v=element(value);remember(key,v);capture();emit("read",line,column,vars,v.position,-1,-1,v.value);return v.value;}
int scan(int &value,int line,int column,Variables vars){Observed v=element(value);capture();emit("scan",line,column,vars,v.position,-1,-1,v.value);return v.value;}
int save(int &value,int key,int line,int column,Variables vars){Observed v=element(value);remember(key,v);saved.push_back({frame,key,v.position,-1,v.value,v.id});capture();emit("save",line,column,vars,v.position,-1,-1,v.value);return v.value;}
Saved &saving(int key){for(auto &s:saved)if(s.frame==frame&&s.key==key)return s;fail("missing saved element");return saved[0];}
int &shift(int &to,int &from,int key,int line,int column,Variables vars){int l=slot(to),r=slot(from);auto &s=saving(key);if(l!=s.left||s.right!=-1)fail("manual exchange mismatch");s.right=r;int x=to,y=from;capture();to=from;ids[l]=ids[r];emit("shift",line,column,vars,l,r,-1,x,y);return to;}
int &drop(int &to,int value,int key,int line,int column,Variables vars){int r=slot(to);auto s=saving(key);if(r!=s.right||value!=s.value)fail("manual exchange mismatch");capture();to=value;ids[r]=s.id;emit("drop",line,column,vars,s.left,r,-1,value);saved.erase(std::remove_if(saved.begin(),saved.end(),[&](const Saved &v){return v.frame==frame&&v.key==key;}),saved.end());return to;}
int &write(int &to,int value,int line,int column,Variables vars){int p=slot(to);capture();int old=to;to=value;emit("write",line,column,vars,p,-1,-1,value,old);return to;}
int *rotate(int *first,int *middle,int *last,int line,int column,Variables vars){int l=boundary(first),m=boundary(middle),r=boundary(last);if(l>m||m>r)fail("invalid rotate range");capture();int *result=std::rotate(first,middle,last);std::rotate(ids+l,ids+m,ids+r);emit("rotate",line,column,vars,l,r,m);return result;}
template<unsigned N> int bucket(int (&values)[N],int index,int delta,bool post,int line,int column,Variables vars){if(index<0||index>=static_cast<int>(N))fail("bucket out of range");int old=values[index];values[index]+=delta;capture();emit("bucket",line,column,vars,-1,-1,-1,delta,old,"",-1,index,values[index]);return post?old:values[index];}
void complete(){if(!saved.empty())fail("unfinished manual exchange");capture();emit("finish",1,1,{});if(std::fclose(file))fail("trace file close");file=nullptr;}
}
