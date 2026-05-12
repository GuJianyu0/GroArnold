//// main
extern "C"{
#include"F_to_C.h"
}
#include"adjust_foci.h"
#include"potential_wrapper.h"
#include<time.h>
// #include<mpi.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <iostream>
// using namespace std;
using namespace UTILITIES;
using namespace SCFORB;

// const double r_range_min = 5e-1; //1e1 nan while 1e1+0.1 not
// const double r_range_max = 2.e2;
// const int N_grid = 32; //used for foci
const int N_grid_x = 100;
const int N_grid_y = 100;
// const int N_grid_z = 100; //to intepolation
const int N_grid_z = 9; //to display
const double bd_kpc = 30.; // |x|, |y|, |z| <= bd_kpc
// const double bd_kpc = 100.;

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::fprintf(stderr,
                     "Usage: %s snapshot_ID (e.g. 10 for snapshot_10)\n",
                     argv[0]);
        return 1;
    }

    // snapshot_ID currently only used to name the output file;
    // SCF coefficients should have been prepared beforehand.
    int snapshot_ID = std::atoi(argv[1]);
    if (snapshot_ID <= 0) {
        std::fprintf(stderr,
                     "ERROR: snapshot_ID must be positive integer, got '%s'\n",
                     argv[1]);
        return 1;
    }

    // File name consistent with plot_action_figs.py:
    // e.g. potential_contour_data_10.txt
    std::string potential_contour_file =
        "potential_contour_data_" + std::to_string(snapshot_ID) + ".txt";

    // Initialise SCF parameters / coefficients
    SCFORB::get_parameter_();

    const int Nx = N_grid_x;
    const int Ny = N_grid_y;
    const int Nz = N_grid_z;

    const double x_min = -bd_kpc;
    const double x_max =  bd_kpc;
    const double y_min = -bd_kpc;
    const double y_max =  bd_kpc;
    const double z_min = -bd_kpc;
    const double z_max =  bd_kpc;

    const double dx = (x_max - x_min) / (Nx - 1);
    const double dy = (y_max - y_min) / (Ny - 1);
    const double dz = (z_max - z_min) / (Nz - 1);

    std::FILE *fp = std::fopen(potential_contour_file.c_str(), "w");
    if (!fp) {
        std::perror("fopen");
        std::fprintf(stderr,
                     "ERROR: cannot open output file '%s'\n",
                     potential_contour_file.c_str());
        return 1;
    }

    std::vector<double> pos(3);
    std::size_t n_written = 0;

    for (int ix = 0; ix < Nx; ++ix) {
        double x = x_min + dx * ix;
        for (int iy = 0; iy < Ny; ++iy) {
            double y = y_min + dy * iy;
            for (int iz = 0; iz < Nz; ++iz) {
                double z = z_min + dz * iz;

                pos[0] = x;
                pos[1] = y;
                pos[2] = z;

                // SCF potential at (x,y,z)
                double pot = potential_SCF(pos);

                // One line: x [kpc]  y [kpc]  z [kpc]  Phi [(km/s)^2]
                std::fprintf(fp, "%.16e %.16e %.16e %.16e\n", x, y, z, pot);
                ++n_written;
            }
        }
    }

    std::fclose(fp);

    std::cout << "Wrote SCF potential grid to '" << potential_contour_file
              << "' with " << n_written << " grid points "
              << "(" << Nx << " x " << Ny << " x " << Nz << ").\n";

    return 0;
}
