/**
 * \file osconfig.h
 * \brief Operating System configuration file.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */
#ifndef FLOWFILTER_OSCONFIG_H_
#define FLOWFILTER_OSCONFIG_H_


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