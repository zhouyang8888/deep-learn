#ifndef __block__
#define __block__

class block 
{
public:
    virtual bool operator<(const block& o) const =0;
    virtual bool operator<=(const block& o) const =0;
    virtual bool operator==(const block& o) const =0;

    virtual void dump() =0; 
};

#endif
