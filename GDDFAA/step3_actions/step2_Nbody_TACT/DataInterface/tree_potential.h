// tree_potential.h
#ifndef TREE_POTENTIAL_H
#define TREE_POTENTIAL_H

#include <vector>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>

// Self-contained octree (Barnes–Hut) module that computes gravitational potential only.
// - Per-particle softening length.
// - Gadget-like spline softening kernel (same form as in your Gadget2FormatData_io.cpp).
// - No dependency on MPI / Gadget structs / kdtree / Eigen / legacy headers.
//
// Usage pattern (intended):
//   1) Build once per snapshot: tree.build(...)
//   2) Query many times: tree.potential(x,y,z, eps_target, exclude_original_id)
//
// Notes:
//   - Softening interaction length is max(eps_source, eps_target) if eps_target>0.
//   - Opening criterion uses geometric center for acceptance, monopole at COM for contribution.
//   - If you want “test particle” convention, pass eps_target=0.

namespace treegrav {

class OctreePotential {
public:
    struct Particle {
        double x, y, z;
        double m;
        double eps;   // softening length (epsilon); internally uses h = eps * 2.8
    };

    explicit OctreePotential(double G = 1.0, double theta = 0.7, int bucket_size = 16, int max_depth = 32);

    void clear();

    // Build from AoS. The caller controls how indices map:
    // - If your particle storage is 1-based P[1..N], pass pointer as P and first_index=1.
    // - If you pass pointer as (P+1), pass first_index=0.
    void build(const Particle* p, int n, int first_index = 0, bool reorder_particles = true);

    // Build from SoA. Same first_index rules: arrays must be valid at [first_index .. first_index+n-1].
    void build(const double* x, const double* y, const double* z,
               const double* m, const double* eps,
               int n, int first_index = 0, bool reorder_particles = true);

    // Potential at point (x,y,z).
    // eps_target:
    //   - 0: use source-only eps (matches your potential_sum(x_tgt) style).
    //   - >0: interaction softening = max(eps_source, eps_target) for leaf particles,
    //         and max(node.eps_max, eps_target) for monopoles.
    // exclude_original_id:
    //   - pass an original particle ID to exclude its contribution (e.g. for “self” potential removal).
    //   - original_id is interpreted with the same first_index you used in build().
    inline double potential(double x, double y, double z,
                            double eps_target = 0.0,
                            int exclude_original_id = -1) const
    {
        if (n_ <= 0 || nodes_.empty()) return 0.0;

        int exclude_internal = -1;
        if (exclude_original_id >= first_index_ && exclude_original_id < first_index_ + n_) {
            const int old0 = exclude_original_id - first_index_;
            exclude_internal = (particles_reordered_ ? perm_inv_[old0] : old0);
        }

        const double tiny = 1e-30;

        // Fixed stack (fast path). Fallback to dynamic only if needed.
        const int kStackCap = 4096;
        int stack_fixed[kStackCap];
        int top = 0;
        stack_fixed[top++] = 0;

        double pot_acc = 0.0;

        while (top > 0) {
            const int nid = stack_fixed[--top];
            const Node& node = nodes_[nid];

            if (node.mass <= 0.0) continue;

            // acceptance criterion uses distance to node geometric center
            const double dx_c = x - node.cx;
            const double dy_c = y - node.cy;
            const double dz_c = z - node.cz;
            const double r2_c = dx_c*dx_c + dy_c*dy_c + dz_c*dz_c + tiny;

            if (node.is_leaf) {
                pot_acc += eval_leaf_(node, x, y, z, eps_target, exclude_internal);
                continue;
            }

            const double s = 2.0 * node.hs;     // node side length
            const double s2 = s * s;

            if (s2 < theta2_ * r2_c) {
                // accept monopole at COM
                const double dx = x - node.comx;
                const double dy = y - node.comy;
                const double dz = z - node.comz;
                const double r2 = dx*dx + dy*dy + dz*dz + tiny;
                const double r  = std::sqrt(r2);

                double eps_use = node.eps_max;
                if (eps_target > 0.0 && eps_target > eps_use) eps_use = eps_target;

                pot_acc += node.mass * kernel_potential_(r, eps_use);
            } else {
                // open node
                for (int c = 0; c < 8; ++c) {
                    const int ch = node.child[c];
                    if (ch >= 0) {
                        if (top < kStackCap) {
                            stack_fixed[top++] = ch;
                        } else {
                            // Rare overflow fallback: traverse remaining with dynamic stack.
                            std::vector<int> st;
                            st.reserve(1024);
                            st.push_back(ch);
                            for (int cc = c + 1; cc < 8; ++cc) {
                                const int ch2 = node.child[cc];
                                if (ch2 >= 0) st.push_back(ch2);
                            }
                            while (top > 0) st.push_back(stack_fixed[--top]);

                            while (!st.empty()) {
                                const int nid2 = st.back();
                                st.pop_back();
                                const Node& nd = nodes_[nid2];
                                if (nd.mass <= 0.0) continue;

                                const double dx2c = x - nd.cx;
                                const double dy2c = y - nd.cy;
                                const double dz2c = z - nd.cz;
                                const double r22c = dx2c*dx2c + dy2c*dy2c + dz2c*dz2c + tiny;

                                if (nd.is_leaf) {
                                    pot_acc += eval_leaf_(nd, x, y, z, eps_target, exclude_internal);
                                    continue;
                                }

                                const double ss  = 2.0 * nd.hs;
                                const double ss2 = ss * ss;

                                if (ss2 < theta2_ * r22c) {
                                    const double dxm = x - nd.comx;
                                    const double dym = y - nd.comy;
                                    const double dzm = z - nd.comz;
                                    const double r2m = dxm*dxm + dym*dym + dzm*dzm + tiny;
                                    const double rm  = std::sqrt(r2m);

                                    double eps_use2 = nd.eps_max;
                                    if (eps_target > 0.0 && eps_target > eps_use2) eps_use2 = eps_target;

                                    pot_acc += nd.mass * kernel_potential_(rm, eps_use2);
                                } else {
                                    for (int k = 0; k < 8; ++k) {
                                        const int chh = nd.child[k];
                                        if (chh >= 0) st.push_back(chh);
                                    }
                                }
                            }

                            return G_ * pot_acc;
                        }
                    }
                }
            }
        }

        return G_ * pot_acc;
    }

    inline double potential(const double pos[3],
                            double eps_target = 0.0,
                            int exclude_original_id = -1) const
    {
        return potential(pos[0], pos[1], pos[2], eps_target, exclude_original_id);
    }

    int particle_count() const { return n_; }
    std::size_t node_count() const { return nodes_.size(); }

    void set_theta(double theta);
    void set_gravitational_constant(double G) { G_ = G; }

    // Mapping original->internal (useful if you need to exclude a particle efficiently outside).
    // Returns -1 if not valid.
    int internal_index_from_original(int original_id) const;

private:
    struct Node {
        double cx, cy, cz;     // geometric center of cube
        double hs;             // half-size of cube
        double mass;           // total mass
        double comx, comy, comz; // center of mass
        double eps_max;        // max softening in this node
        int child[8];          // indices of children, -1 if none
        int start, end;        // particle range (in perm ordering or reordered arrays)
        bool is_leaf;

        Node()
            : cx(0), cy(0), cz(0), hs(0),
              mass(0), comx(0), comy(0), comz(0),
              eps_max(0), start(0), end(0), is_leaf(false)
        {
            for (int i = 0; i < 8; ++i) child[i] = -1;
        }
    };

    // Particle arrays (SoA) for fast evaluation
    std::vector<double> x_, y_, z_, m_, eps_;

    // If reorder_particles==false, leaves traverse via perm_[k] -> particle index
    std::vector<int> perm_;
    std::vector<int> perm_tmp_;

    // If reorder_particles==true, we reorder (x_,y_,z_,m_,eps_) into perm-order and clear perm_.
    // For exclusion, keep inverse permutation old->new.
    std::vector<int> perm_inv_;
    bool particles_reordered_ = false;

    std::vector<Node> nodes_;

    int n_ = 0;
    int first_index_ = 0;

    double G_ = 1.0;
    double theta_ = 0.7;
    double theta2_ = 0.49;

    int bucket_size_ = 16;
    int max_depth_ = 32;

    // build helpers
    void build_from_arrays_(int n, int first_index, bool reorder_particles);
    void build_node_(int node_id, int depth);

    // kernel helpers (Gadget-like spline potential; same polynomial form you already use)
    static inline double kernel_potential_(double r, double eps)
    {
        const double tiny = 1e-30;
        if (r <= 0.0) r = std::sqrt(tiny);

        if (eps <= 0.0) {
            return -1.0 / (r + tiny);
        }

        const double h = eps * 2.8;
        if (r >= h) {
            return -1.0 / (r + tiny); // exact Newtonian outside softening radius
        }

        const double inv_h = 1.0 / h;
        const double u = r * inv_h;

        double W;
        if (u < 0.5) {
            // W(u) = -2.8 + u^2*(5.333333333333 + u^2*(6.4*u - 9.6))
            const double u2 = u * u;
            W = -2.8 + u2 * (5.333333333333 + u2 * (6.4 * u - 9.6));
        } else {
            // W(u) = -3.2 + 0.066666666667/u + u^2*(10.666666666667 + u*(-16 + u*(9.6 - 2.133333333333*u)))
            const double inv_u = 1.0 / (u + tiny);
            const double u2 = u * u;
            W = -3.2 + 0.066666666667 * inv_u
                + u2 * (10.666666666667 + u * (-16.0 + u * (9.6 - 2.133333333333 * u)));
        }

        return inv_h * W;
    }

    inline double eval_leaf_(const Node& node,
                             double tx, double ty, double tz,
                             double eps_target,
                             int exclude_internal) const
    {
        const double tiny = 1e-30;
        double acc = 0.0;

        if (perm_.empty()) {
            // particles are reordered: [start,end) directly indexes x_/y_/z_/m_/eps_
            for (int i = node.start; i < node.end; ++i) {
                if (i == exclude_internal) continue;
                const double dx = tx - x_[i];
                const double dy = ty - y_[i];
                const double dz = tz - z_[i];
                const double r2 = dx*dx + dy*dy + dz*dz + tiny;
                const double r  = std::sqrt(r2);

                double eps_use = eps_[i];
                if (eps_target > 0.0 && eps_target > eps_use) eps_use = eps_target;

                acc += m_[i] * kernel_potential_(r, eps_use);
            }
        } else {
            // particles not reordered: perm_[k] gives particle index into arrays
            for (int k = node.start; k < node.end; ++k) {
                const int i = perm_[k];
                if (i == exclude_internal) continue;
                const double dx = tx - x_[i];
                const double dy = ty - y_[i];
                const double dz = tz - z_[i];
                const double r2 = dx*dx + dy*dy + dz*dz + tiny;
                const double r  = std::sqrt(r2);

                double eps_use = eps_[i];
                if (eps_target > 0.0 && eps_target > eps_use) eps_use = eps_target;

                acc += m_[i] * kernel_potential_(r, eps_use);
            }
        }

        return acc;
    }
};

} // namespace treegrav

#endif // TREE_POTENTIAL_H
