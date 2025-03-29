//Keirn Hally

//2201071@uad.ac.uk

//GPU parrallel method for completing Pascals Triangle



#include <CL/sycl.hpp>//sycl header
#include <iostream>// input output stream
#include <vector>// vector container



using namespace sycl;// Access sycl computation
constexpr size_t local_size = 4096;  // Local size chosen based on typical GPU capabilities



int main() {

    try {

        sycl::queue q(sycl::cpu_selector{}, property_list{});
        std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << "\n";
        constexpr size_t numRows = 2;

        // Allocate Unified Shared Memory for the triangle

        auto triangle = malloc_shared<int*>(numRows, q)

        for (size_t i = 0; i < numRows; ++i) {

            triangle[i] = malloc_shared<int>(i + 1, q);
            //strating position for Pascals triangle

            if (i == 0) {

                triangle[i][0] = 1;

            }

            else {

                q.parallel_for(range<1>(i + 1), [=](id<1> idx) {

                    int index = idx[0];

                    // Boundary

                    if (index == 0 || index == i) {



                        triangle[i][index] = 1;  // initialize the boundary elements

                    }

                    }).wait();
            }

        }

        // Compute Pascal's Triangle using both USM and local memory

        for (size_t i = 2; i < numRows; ++i) {
            
            buffer<int, 1> bufCurr(triangle[i], range<1>(i + 1));

            q.submit([&](handler& h) {

                auto accCurr = bufCurr.get_access<access::mode::read_write>(h);

                accessor<int, 1, access::mode::read_write, access::target::local> localPrev(range<1>(local_size + 2), h);

                // Access kenral for parallel computation across multiple units

                h.parallel_for(nd_range<1>{range<1>(i + 1), range<1>(local_size)}, [=](nd_item<1> it) {



                    size_t local_id = it.get_local_id(0);//get local id of work item within work group

                    size_t global_id = it.get_global_id(0);// get global id of work item across all working groups



                    if (local_id == 0) { // Load one element before the current block if not the first



                        if (global_id > 0) localPrev[0] = triangle[i - 1][global_id - 1];

                    }

                    if (global_id < i) {

                        localPrev[local_id + 1] = triangle[i - 1][global_id];

                    }

                    it.barrier();

                    // Compute the current element using the local memory

                    if (global_id > 0 && global_id < i) {

                        accCurr[global_id] = localPrev[local_id] + localPrev[local_id + 1];

                    }

                    });

                }).wait();

        }

        // Itterate through the rows and triangle data to output Pascals triangle

        for (size_t i = 0; i < numRows; ++i) {

            for (size_t j = 0; j <= i; ++j) {

                std::cout << triangle[i][j] << " ";

            }

            std::cout << std::endl;

            free(triangle[i], q);  // Free each row

        }


        free(triangle, q);  // Free the array of pointers

    }

    catch (const sycl::exception& e) {

        std::cerr << "SYCL exception caught: " << e.what() << '\n';

        return -1;

    }
}
