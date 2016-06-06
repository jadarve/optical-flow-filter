
#ifndef FLOWFILTER_OSCONFIG_H_
#define FLOWFILTER_OSCONFIG_H_


// #define XSTR(x) STR(x)
// #define STR(x) #x

// if compilling in a Windows environment
#if defined(_WIN32)

	#ifdef FLOWFILTERLIBRARY_EXPORTS
		#define FLOWFILTER_API __declspec(dllexport)
	#else
		#define FLOWFILTER_API __declspec(dllimport) 
	#endif

#else
	// empty
	#define FLOWFILTER_API

#endif

#endif // FLOWFILTER_OSCONFIG_H_