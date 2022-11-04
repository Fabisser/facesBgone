#pragma once

#include <vector>
#include <functional> // for std::reference_wrapper<T>
#include <future> // for std::async
#include <mutex> // for std::mutex
#include <chrono> // for Timer
#include <thread> // for std::this_thread::sleep_for(seconds(5));

#include "Polyhedron.hpp"
#include "JsonHandler.hpp"


using namespace std::chrono;


// Timer class -> used for tracking the run time
//TODO
//gcc compiler doesn't have overloading of "=" for std::chrono::high_resolution_clock::now()
//if compiled using gcc/g++ this might need to be changed
//struct Timer //for counting the time
//{
//  std::chrono::time_point<std::chrono::steady_clock>start, end;
//  std::chrono::duration<float>duration;
//
//  Timer() //set default value
//  {
//	start = std::chrono::high_resolution_clock::now();
//	end = std::chrono::high_resolution_clock::now();
//	duration = end - start;
//  }
//
//  ~Timer() // get the end value and print the duration time
//  {
//	end = std::chrono::high_resolution_clock::now();
//	duration = end - start;
//
//	std::cout << "Time: " << duration.count() << "s\n";
//  }
//};



/*
* namespace MT -> stands for multi threading
* the functions inside this namespace are for performing
* multi threading / not multi threading tasks
* function with "sync" means single thread process
* function with "async" means multi threading process
* 
* CGAL does resources management internally
* small primitives such as the components of Nef_polyhedron_3 are sotred in 
* the so-called "Compact_container"
* the memory address of the small primitives could be changed automatically
* thus handlers are used for accessing these primitives
* and pointers should be avoided as much as possible when using CGAL
* 
* pass the CGAL object as function parameters with references
* for std::async(), pass the pointers whenever possible?
* 
* since building Nef_polyhedron is relatively fast
* we build the nefs first and then multi thread the minkowski sum process
*/
namespace MT {


//std::vector<std::future<void>> futures; // store the return value of std::async, necessary step to make async work
//std::mutex nef_mutex; // for thread-safety


/*
* use minkowski sum to expand a nef
* add the expanded nef to expanded_nefs vector (via pointer)
*
* @ param:
*
* @ nef:
* nef which will be expanded, it's a CGAL object, thus we pass it using reference as a parameter
*
* @ expanded_nefs:
* a vector to store the expanded nefs, for using std::async(), we pass the pointer of the vector
*
* @ minkowski_param:
* the "minkowski parameter"
* minkowski sums two nefs, we define a small cube with side length = minkowski_param
* and we expand the nef with this cube
* the minkowski_param is set to 0.1 by default
*/
//void expand_nef_async(
//	Nef_polyhedron& nef,
//	std::vector<Nef_polyhedron>* expanded_nefs_Ptr,
//	double minkowski_param)
//{
//  // check the pointer
//  if (expanded_nefs_Ptr == nullptr) {
//	std::cerr << "pointer of expanded_nefs vector is null, please check " << std::endl;
//	return;
//  }
//
//  // perform minkowski operation
//  //TODO
//  //try/catch with mutex ... ?
//  try{
//	Nef_polyhedron expanded_nef = NefProcessing::minkowski_sum(nef, minkowski_param);
//	std::lock_guard<std::mutex> lock(nef_mutex); // lock the meshes to avoid conflict
//	expanded_nefs_Ptr->emplace_back(expanded_nef);
//  }catch(CGAL::Assertion_exception e){
//	// inside catch can not process the nef
//	std::cerr << "CGAL error" << '\n';
//	std::cout << "the nef will be skipped\n";
//  }

//  Nef_polyhedron expanded_nef = NefProcessing::minkowski_sum(nef, minkowski_param);
//
//  // using a local lock_guard to lock mtx guarantees unlocking on destruction / exception:
//  std::lock_guard<std::mutex> lock(nef_mutex); // lock the meshes to avoid conflict
//  expanded_nefs_Ptr->emplace_back(expanded_nef);



/*
* expand nefs asynchronously
* will call std::async() and expand_nef_async() function
*
* @ param:
*
* @ nefs:
* a vector containing all the original nef polyhedra
*
* @ expanded_nefs:
* a vector which contains the expanded nef polyhedra
* to use std::async() we need to pass pointers whenever possible
* but we need to avoid to use the pointer pointing to CGAL objects
*
* @ minkowski_param:
* the "minkowski parameter"
* minkowski sums two nefs, we define a small cube with side length = minkowski_param
* and we expand the nef with this cube
* the minkowski_param is set to 0.1 by default
*/
//void expand_nefs_async(
//	std::vector<Nef_polyhedron>& nefs,
//	std::vector<Nef_polyhedron>& expanded_nefs,
//	double minkowski_param = 0.1)
//{
//
//  /*
//  * call std::async() to enable asynchronous process
//  * it is important to save the result of std::async()
//  * to enable the async process
//  *
//  * do not use const qualifier - the nef will be changed
//  * and use reference in the for loop
//  */
//  for (auto& nef : nefs) {
//	//auto futureobj = std::async(std::launch::async, expand_nef_async, nef, &expanded_nefs, minkowski_param);
//	futures.emplace_back(
//		std::async(
//			std::launch::async, /* launch policy */
//			expand_nef_async, /* function will be called asynchronously */
//			nef, /* arguments - a nef */
//			&expanded_nefs, /* arguments - pointer to expanded_nefs vector */
//			minkowski_param /* arguments - minkowski_param (default is 0.1)*/
//		));
//  }
//
//  /*
//  * if we wish to get the result value and keep processing
//  * we need to use get() of every future object
//  */
//  for (auto& futureObject : futures) {
//	futureObject.get();
//  }
//}



/* ----------------------------------------------------------------------------------------------------------------*/



/*
* expand a nef
* it's the synchronous version of expand_nef_async()
* used for not performing multi threading
*/
void expand_nef(
	Nef_polyhedron& nef,
	std::vector<Nef_polyhedron>* expanded_nefs_Ptr,
	double minkowski_param)
{
  std::cout << "expand_nef\n";
  // check the pointer
  if (expanded_nefs_Ptr == nullptr) {
	std::cerr << "pointer of expanded_nefs vector is null, please check " << std::endl;
	return;
  }

  // before peroforming minkowski operation, make a copy of nef
  // since if an exception was shrown, it is not possible to use nef in the catch block
  Nef_polyhedron nef_copy(nef);
  std::cout << "make copy\n";

  // perform minkowski operation
  try{
	Nef_polyhedron expanded_nef = NefProcessing::minkowski_sum(nef, minkowski_param);
	expanded_nefs_Ptr->emplace_back(expanded_nef);
  }catch(...){

	// inside catch can not process the nef
	std::cerr << "CGAL error" << '\n';

	// first get the geometry
	std::vector<Shell_explorer> shell_explorers;
	NefProcessing::extract_nef_geometries(nef_copy, shell_explorers);

	Shell_explorer se = shell_explorers[0]; // shell representing for exterior
	Polyhedron convex_polyhedron;
	CGAL::convex_hull_3(se.vertices.begin(), se.vertices.end(), convex_polyhedron);

	if(convex_polyhedron.is_closed()){
	  Nef_polyhedron convex_nef(convex_polyhedron);
	  Nef_polyhedron expanded_convex_nef = NefProcessing::minkowski_sum(convex_nef, minkowski_param);
	  expanded_nefs_Ptr->emplace_back(expanded_convex_nef);
	  std::cout << "build the convex hull of the nef and then expand\n";
	}else{
	  std::cout << "the nef will be skipped\n";
	}
  }

  std::cout << "done\n";

}


/*
* expand nefs
* it's the synchronous version of expand_nefs_async()
* used for not performing multi threading
*/
void expand_nefs(
	std::vector<Nef_polyhedron>& nefs,
	std::vector<Nef_polyhedron>& expanded_nefs,
	double minkowski_param = 0.1)
{
  // expand each nef in nefs vector
  for (auto& nef : nefs) {
	try{
	  expand_nef(nef, &expanded_nefs, minkowski_param);
	}catch(...){
	  std::cerr << "expand nef error\n";
	  continue;
	}

  }
}

};