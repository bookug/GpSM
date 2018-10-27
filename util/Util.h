/*=============================================================================
# Filename: util.h
# Author: Bookug Lobert 
# Mail: 1181955272@qq.com
# Last Modified: 2016-10-24 17:20
# Description: 
=============================================================================*/

#ifndef _UTIL_UTIL_H
#define _UTIL_UTIL_H

//#ifdef _WIN32
//#   pragma warning( disable : 4996 ) // disable deprecated warning 
//#endif

//#ifdef __cplusplus
//extern "C" {
//#endif

//basic macros and types are defined here, including common headers 

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <dirent.h>
#include <string.h>
#include <unistd.h>
#include <ctype.h>
#include <time.h>
#include <fcntl.h>
#include <errno.h>
#include <regex.h>
#include <locale.h>
#include <assert.h>
#include <libgen.h>

#include <sys/time.h>
#include <sys/types.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/mman.h>

#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>

//NOTICE:below are restricted to C++, C files should not include(maybe nested) this header!
#include <bitset>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

#include <map>
#include <set>
#include <stack>
#include <queue>
#include <vector>
#include <list>
#include <iterator>
#include <algorithm>
#include <functional>
#include <utility>

//NOTICE:below are libraries need to link
#include <math.h>
#include <readline/readline.h>
#include <readline/history.h>

//indicate that in debug mode
//#define DEBUG   1

#define MAX_PATTERN_SIZE 1000000000


#define xfree(x) free(x); x = NULL;

typedef int LABEL;
typedef int VID;
typedef int EID;
typedef int GID;
typedef long PID;
typedef long LENGTH;

static const unsigned INVALID = UINT_MAX;

/******** all static&universal constants and fucntions ********/
class Util
{
public:
	Util();
	~Util();
	static long get_cur_time();
};

//#ifdef __cplusplus
//}
//#endif  // #ifdef _DEBUG (else branch)

#endif //_UTIL_UTIL_H

