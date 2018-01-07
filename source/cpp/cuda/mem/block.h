#ifndef __block__
#define __block__

struct block 
{
    virtual bool operator<(const block& o) const =0;
    virtual bool operator<=(const block& o) const =0;
    virtual bool operator==(const block& o) const =0;

    virtual const void dump() const =0; 
};

#endif
