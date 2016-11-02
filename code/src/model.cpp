//#include "model.h"
#include "../include/model.h"


void Model::initmodel(const string& modelfile)
{
    fstream filein;
    filein.open(modelfile.c_str(), ios::in|ios::binary);
    initmodel(filein);
    filein.close();
}
void Model::initmodel(fstream& filein)
{
    std::cout<<"read model"<<std::endl;
    opts.Init(filein);
    thrs.Init(filein);
//    thrs.Save("thrs", false);

    fids.Init(filein);
//    fids.Save("fids", true);

    child.Init(filein);
//    child.Save("child", true);

    count.Init(filein);
//    count.Save("count", true);

    depth.Init(filein);
//    depth.Save("depth", true);

    nSegs.Init(filein);
//    nSegs.Save("nsegs", true);

    eBins.Init(filein);
//    eBins.Save("ebins", true);

    eBnds.Init(filein);
//    eBnds.Save("ebnds", true);

    segs.Init(filein);
//    segs.Save("segs", true);
}
Model::~Model()
{
}
