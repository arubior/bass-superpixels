#ifndef _DB_FEATURE_
#define _DB_FEATURE_

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>

template <typename T>
class dbFeature
{
public:
    dbFeature()
    {
        m_nNum = 0;
        m_nDim = 0;
        m_pData = NULL;
    }

    ~dbFeature()
    {
        clear();
    }

    void clear()
    {
        if(m_pData != NULL)
        {
            delete[] m_pData;
            m_pData = NULL;
        }
        m_nNum = 0;
        m_nDim = 0;
    }

public:
    void test_print_item(int idx)
    {
        if(idx < 0 || idx >= m_nNum)
        {
            printf("idx out of range!\n");
            return;
        }

        printf("test %d th item in dbFeature...\n", idx);
        printf("feature first 10 dim:\n");
        const T* pf = m_pData + idx * m_nDim;
        for(int i = 0; i < 10; i++)
        {
            printf("%.2f\t", *pf++);
        }
        printf("\n");


    }

public:
    void set_data(const T* pData, int nNum, int nDim)
    {
        clear();

        m_pData = new T[nNum * nDim];
        memcpy(m_pData, pData, sizeof(T)*nNum*nDim);
        m_nNum = nNum;
        m_nDim = nDim;
    }

    void set_size(int nNum, int nDim)
    {
        clear();
        m_pData = new T[nNum * nDim];
        memset(m_pData, 0, sizeof(T) * nNum * nDim);
        m_nNum = nNum;
        m_nDim = nDim;
    }

    void set_data(int idx, const T* pData, int len = 1)
    {
        assert(idx >=0 && idx + len <= m_nNum);
        assert(m_pData != NULL);

        T* pTo = m_pData + idx * m_nDim;
        memcpy(pTo, pData, sizeof(T) * len * m_nDim);
    }

    int get_number() const
    {
        return m_nNum;
    }

    int get_dim() const
    {
        return m_nDim;
    }

    T* get_data_pointer(int idx = 0) const
    {
        if(idx < 0 || idx >= m_nNum)
        {
            return NULL;
        }
        return m_pData + idx * m_nDim;
    }

public:
    bool save_to_file(const char* file) const //binary file
    {
        FILE* fp = fopen(file, "wb");
        if(fp == NULL)
        {
            return false;
        }
        fwrite(&m_nNum, sizeof(int), 1, fp);
        fwrite(&m_nDim, sizeof(int), 1, fp);
        fwrite(m_pData, sizeof(T), m_nNum * m_nDim, fp);

        fflush(fp);
        fclose(fp);
        return true;
    }

    bool read_from_file(const char* file) //binary file
    {
        FILE* fp = fopen(file, "rb");
        if(fp == NULL)
        {
            return false;
        }

        clear();

        fread(&m_nNum, sizeof(int), 1, fp);
        fread(&m_nDim, sizeof(int), 1, fp);
        m_pData = new T[m_nNum *m_nDim];
        fread(m_pData, sizeof(T), m_nNum * m_nDim, fp);
        fclose(fp);
        return true;
    }

    bool save_to_text_file(const char* file) const
    { //text file
        std::ofstream fout;
        fout.open(file);
        if(fout.fail())
        {
            return false;
        }
        fout << m_nNum << std::endl;
        fout << m_nDim << std::endl;
        const T *data = m_pData;
        for(int i=0; i<m_nNum; i++)
        {
            for(int j=0; j<m_nDim; j++)
            {
                fout << *(data++) << " ";
            }
            fout << std::endl;
        }
        fout.flush();
        fout.close();
        return true;
    }

    bool read_from_text_file(const char* file)
    {
        std::ifstream fin;
        fin.open(file);
        if(fin.fail()) return false;
        fin >> m_nNum;
        fin >> m_nDim;
                printf("%d,%d\n", m_nDim,m_nNum);

        m_pData = new T[m_nNum *m_nDim];
        T *data = m_pData;
        for(int i=0; i<m_nNum; i++)
            for(int j=0; j<m_nDim; j++)
                fin >> *(data++);
        fin.close();
        return true;
    }


private:
    int m_nNum;
    int m_nDim;
    T* m_pData;
};

#endif
