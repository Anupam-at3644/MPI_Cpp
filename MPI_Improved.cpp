/*Message Passing Interface
-> Used for distributed computing, i.e., across multiple pairs of processor and memory that are part of a communicating network
-> Supported in multiple languages and different platforms
-> Implemented mostly through MPICH or Open-MPI (Open source libraries)
-> Processes need to be run on multiple systems and they communicate through a "communicator"
-> This "communicator" is encapsulated by MPI in the type "MPI_Comm" (default: MPI_COMM_WORLD, which involves all systems part of the network)
-> Important Skeletal Functions:
    -> Initialize                                            - int MPI_Init (int* argc, char*** argv);
    -> Close (Close communication channels/ Free up memory)  - int MPI_Finalize();
    -> All processes                                         - int MPI_Comm_size(MPI_Comm comm, int* size);
    -> Local index of process                                - int MPI_Comm_rank(MPI_Comm comm, int* rank);

->Typical Structure:
int main(int argc, char** argv)
{
    int num_procs;
    int rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    //Parallel tasks// 
    
    MPI_Finalize();
}
    
    
*/

#include <iostream>
#include <mpi.h>
#include <vector>
#include <numeric>
#include <cmath>
using namespace std;

int main(int argc, char** argv)
{
    // Declare total_ranks and my_rank -> index of individual processor
    int total_ranks;
    int my_rank;


    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &total_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);


    // Create random number of elements in each process with random values in range 0-180 (theta)
    srand(my_rank + time(NULL));
    int num_elements = rand() % 10;


    // All processes create their own elements stored in original_array
    vector<int> original_array;
    int theta;
    for (int i = 0; i < num_elements; i++)
    {
        theta = rand() % 181;
        original_array.push_back(theta);
    }


    // Print statements to check the initialized values in each process
    printf("\nINITIALIZE %d:    Hello!, I am rank (processor index) %d of a total of %d processors with %d elements. My elements are:",
        my_rank, my_rank, total_ranks, num_elements);
    for (int i = 0; i < num_elements; i++)
    {
        printf("%d ", original_array[i]);
    }
    

    int total_elements;
    // Collect all num_elements at master rank (assumed as rank 0)
    vector<int> number_of_elements_array(total_ranks);  // buffer to store gathered information from all processes
    MPI_Gather(&num_elements, 1, MPI_INT, number_of_elements_array.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);


    if (my_rank == 0)
    {
        // Sum all members of number_of_elements_array to calculate total number of elements of interest (needed later)
        total_elements = accumulate(number_of_elements_array.begin(), number_of_elements_array.end(), 0);
        printf("\n\nTotal elements are %d", total_elements);
    }


    // Displacements_array (array of index numbers where to store information from each process into buffer array)
    // For ex: for number of elements to be sent array as {a, b, c, ...}
    // The displacements will  be {0, a, a+b, a+b+c, ...} 
    // These are needed for MPI_Gatherv/MPI_Scatterv function parameters
    vector<int> displacements_array_2;
    

    if  (my_rank == 0)
    {
        // Intialize new array with 0 (first index of buffer array)
        displacements_array_2.push_back(0);

        for (int i = 0; i < total_ranks - 1; i++)
        {
            displacements_array_2.push_back(displacements_array_2[i] + number_of_elements_array[i]);
        }
    }


    // Create new buffer to store all elements *sequentially* from all processes in a single array
    // For processes {A, B, C, ...}
    // The stored elements array will be {a_1, a_2, ..., b_1, b_2, ..., c_1, c_2, ..., ...}
    vector<int> combined_task_array(total_elements);
    // Collect individual elements from all processes sequentially into a single array
    MPI_Gatherv(original_array.data(), num_elements, MPI_INT,
        combined_task_array.data(), number_of_elements_array.data(), displacements_array_2.data(), MPI_INT, 0, MPI_COMM_WORLD);
    vector<int> redistributed_number_of_elements_array(total_ranks);
    
    MPI_Barrier(MPI_COMM_WORLD);

    if (my_rank == 0)
    {    
        printf("\n\nPROGRESS %d:    The number of elements in processes 0 to %d are: ", my_rank, total_ranks - 1);
        for (int i = 0; i < total_ranks; i++)
        {
            printf("%d ", number_of_elements_array[i]);
        }

        printf("\nPROGRESS %d:    The sequential array of all elements is (from process 0 to %d): ", my_rank, total_ranks - 1);
        for (int i = 0; i < total_elements; i++)
        {
            printf("%d ", combined_task_array[i]);
        }

        // Next: distribute elements equally via process 0
        // Create array to hold the number of elements in each process after redistribution
        // For ex: if number_of_elements_array is {8, 1, 4, 7} redistribute as redistributed_number_of_elements_array being {5, 5, 5, 5}
        //         if number_of_elements_array is {8, 2, 4, 7} redistribute as redistributed_number_of_elements_array being {6, 5, 5, 5}
        //         if number_of_elements_array is {8, 3, 4, 7} redistribute as redistributed_number_of_elements_array being {6, 6, 5, 5}

        // Redistribution
        int base_avg = total_elements / total_ranks;

        for (int i = 0; i < total_ranks; i++)
        {
            redistributed_number_of_elements_array[i] = base_avg;
        }

        int remainder = total_elements % total_ranks;

        for (int i = 0; i < remainder; i++)
        {
            redistributed_number_of_elements_array[i] += 1;
        }
    }


    int num_received_tasks;
    // Scatter equalized number of elements that is needed in other processes
    MPI_Scatter(redistributed_number_of_elements_array.data(), 1, MPI_INT, &num_received_tasks, 1, MPI_INT, 0, MPI_COMM_WORLD);


    // Again create displacements_array as a parameter to MPI_Scatterv
    vector<int> displacements_array_3;
    // Declare buffer to store task_array that will have elements on which the task needs to be performed
    vector<int> task_array(num_received_tasks);


    if (my_rank == 0)
    {   
        // Print redistributed array
        printf("\nPROGRESS %d:    The targetted redistribution array is: ", my_rank);
        for (int i = 0; i < total_ranks; i++)
        {
            printf("%d ", redistributed_number_of_elements_array[i]);
        }
    
        // Intialize new array with 0 (first index of buffer task_array)
        displacements_array_3.push_back(0);

        for (int i = 0; i < total_ranks - 1; i++)
        {
            displacements_array_3.push_back(displacements_array_3[i] + redistributed_number_of_elements_array[i]);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    // Redistribute elements equally to perform a task
    MPI_Scatterv(combined_task_array.data(), redistributed_number_of_elements_array.data(), displacements_array_3.data(), MPI_INT, task_array.data(),
        num_received_tasks, MPI_INT, 0, MPI_COMM_WORLD);

    
    // Print statements to check the distributed elements
    if (my_rank == 0)
    {
        printf("\n");
    }
    printf("\nTASK %d:    Hello! I am process %d and my task array is: ", my_rank, my_rank);
    for (int i = 0; i < num_received_tasks; i++)
    {
        printf("%d ", task_array[i]);
    }


    // Perform the task
    vector<float> results_array(num_received_tasks);

    for (int i = 0; i < task_array.size(); i++)
    {
        results_array[i] = sin(task_array[i] * atan(1) / 45.0);
    }
    

    // Gather results
    vector<float> combined_results_array(total_elements);
    MPI_Gatherv(results_array.data(), num_received_tasks, MPI_FLOAT,
        combined_results_array.data(), redistributed_number_of_elements_array.data(), displacements_array_3.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);


    if (my_rank == 0)
    {
        // Print combined results array
        printf("\n\nRESULT %d:    The combined results array is: ", my_rank);
        for (int i = 0; i < total_elements; i++)
        {
            printf("%f ", combined_results_array[i]);
        }
    }


    vector<float> final_results_array(num_elements);
    // Send back results to original processes;
    MPI_Scatterv(combined_results_array.data(), number_of_elements_array.data(), displacements_array_2.data(), MPI_FLOAT, 
        final_results_array.data(), num_elements, MPI_FLOAT, 0, MPI_COMM_WORLD);


    // Print final results
    printf("\nRESULT %d:    Hello! I am process %d and the final results are:", my_rank, my_rank);
    for (int i = 0; i < num_elements; i++)
    {
        printf("%f ", final_results_array[i]);
    }


    MPI_Finalize();

}
