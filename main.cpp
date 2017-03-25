#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <iomanip>
#include "spline.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/KroneckerProduct>
#include "KnotVector.h"

using namespace std;
using namespace spline;
using namespace Eigen;
const double pi = 3.14159265358979323846264338327;
typedef Eigen::SparseVector<double> SpVec;
typedef Eigen::SparseMatrix<double> SpMat;
typedef std::vector<std::pair<double, double>> Span;

void Geometry1(double xi, double eta, double &pxpxi, double &pxpeta, double &pypxi, double &pypeta,
               double &pxpxi_xi, double &pxpxi_eta, double &pxpeta_eta, double &pypxi_xi, double &pypxi_eta,
               double &pypeta_eta,
               double &x, double &y);

void Geometry2(double xi, double eta, double &pxpxi, double &pxpeta, double &pypxi, double &pypeta,
               double &pxpxi_xi, double &pxpxi_eta, double &pxpeta_eta, double &pypxi_xi, double &pypxi_eta,
               double &pypeta_eta,
               double &x, double &y);

double exactSolution(double x, double y) {
    return pow((x - 4) * (y - 4) * x * y, 2);
}

double exactSolution_dx(double x, double y) {
    return 4 * (x - 4) * (x - 2) * x * pow(y * (y - 4), 2);
}

double exactSolution_dy(double x, double y) {
    return 4 * (y - 4) * (y - 2) * y * pow(x * (x - 4), 2);
}

double forceTerm(double x, double y) {
    double s1, s2, s3, s4;
    s1 = 8 * pow(x - 4, 2) * pow(y - 4, 2);
    s2 = 32 * pow(x - 4, 1) * x * pow(y - 4, 2);
    s3 = 8 * x * x * pow(y - 4, 2);
    s4 = 24 * pow(y - 4, 2) * y * y;

    return (s1 + s2 + s3 + s4);
}

int main() {
    double *gaussian = x7;
    double *weight = w7;
    int gaussian_points = 7;
    double epsilon = 1e13;
    KnotVector<double> Xi_patch1, Xi_patch2, Eta_patch1, Eta_patch2, Lagrangian_constaints, Lagrangian_dirichlet_patch1, Lagrangian_dirichlet_patch2;
    unsigned p, refine;
    cin >> p >> refine;
    Xi_patch1.InitClosed(p);
    Xi_patch1.UniformRefine(refine, 1);
    Xi_patch2.InitClosed(p);
    Xi_patch2.UniformRefine(refine, 1);
    Eta_patch1.InitClosed(p);
    Eta_patch1.UniformRefine(refine, 1);
    Eta_patch2.InitClosed(p);
    Eta_patch2.UniformRefine(refine, 1);
    Lagrangian_constaints.InitClosed(p);
    Lagrangian_constaints.UniformRefine(refine, 1);
    Lagrangian_dirichlet_patch1.InitClosed(p);
    Lagrangian_dirichlet_patch1.UniformRefine(refine, 1);
    Lagrangian_dirichlet_patch2.InitClosed(p);
    Lagrangian_dirichlet_patch2.UniformRefine(refine, 1);
    Eta_patch1.printKnotVector();
    Eta_patch2.printKnotVector();
    KnotVector<double> Eta_intersection(Eta_patch1.UniKnotUnion(Eta_patch2));
    const int m_Xi_patch1 = Xi_patch1.GetSize() - 1, m_Xi_patch2 = Xi_patch2.GetSize() - 1, m_Eta_patch1 =
            Eta_patch1.GetSize() - 1, m_Eta_patch2 = Eta_patch2.GetSize() - 1, m_Lagrangian_constaints =
            Lagrangian_constaints.GetSize() - 1, m_Lagrangian_dirichlet_patch1 =
            Lagrangian_dirichlet_patch1.GetSize() - 1, m_Lagrangian_dirichlet_patch2 =
            Lagrangian_dirichlet_patch2.GetSize() - 1;
    const int dof_Xi_patch1 = m_Xi_patch1 - p, dof_Eta_patch1 = m_Eta_patch1 - p, dof_Xi_patch2 =
            m_Xi_patch2 - p, dof_Eta_patch2 = m_Eta_patch2 - p, dof_Lagrangian_constaints =
            m_Lagrangian_constaints - p, dof_Lagrangian_dirichlet_patch1 =
            m_Lagrangian_dirichlet_patch1 - p, dof_Lagrangian_dirichlet_patch2 = m_Lagrangian_dirichlet_patch2 - p;
    const int dof_patch1 = dof_Xi_patch1 * dof_Eta_patch1, dof_patch2 = dof_Xi_patch2 * dof_Eta_patch2;
    vector<double> Xi_patch1_vector(Xi_patch1.GetKnotVector()), Xi_patch2_vector(
            Xi_patch2.GetKnotVector()), Eta_patch1_vector(Eta_patch1.GetKnotVector()), Eta_patch2_vector(
            Eta_patch2.GetKnotVector()), Lagrangian_constaints_vector(
            Lagrangian_constaints.GetKnotVector()), Lagrangian_dirichlet_patch1_vector(
            Lagrangian_dirichlet_patch1.GetKnotVector()), Lagrangian_dirichlet_patch2_vector(
            Lagrangian_dirichlet_patch2.GetKnotVector());
    double *knots_Xi_patch1 = &Xi_patch1_vector[0];
    double *knots_Eta_patch1 = &Eta_patch1_vector[0];
    double *knots_Xi_patch2 = &Xi_patch2_vector[0];
    double *knots_Eta_patch2 = &Eta_patch2_vector[0];
    double *knots_Lagrangian_constaints = &Lagrangian_constaints_vector[0];
    double *knots_Lagrangian_dirichlet_patch1 = &Lagrangian_dirichlet_patch1_vector[0];
    double *knots_Lagrangian_dirichlet_patch2 = &Lagrangian_dirichlet_patch2_vector[0];
    Span Xi_span_patch1(Xi_patch1.KnotSpans());
    Span Xi_span_patch2(Xi_patch2.KnotSpans());
    Span Eta_span_patch1(Eta_patch1.KnotSpans());
    Span Eta_span_patch2(Eta_patch2.KnotSpans());
    Span Eta_span_intersection(Eta_intersection.KnotSpans());

    MatrixXd K = MatrixXd::Zero(
            dof_patch1 + dof_patch2 +
            2 * (dof_Lagrangian_constaints + dof_Lagrangian_dirichlet_patch1 + dof_Lagrangian_dirichlet_patch2),
            dof_patch1 + dof_patch2 +
            2 * (dof_Lagrangian_constaints + dof_Lagrangian_dirichlet_patch1 + dof_Lagrangian_dirichlet_patch2));
    VectorXd F = VectorXd::Zero(
            dof_patch1 + dof_patch2 +
            2 * (dof_Lagrangian_constaints + dof_Lagrangian_dirichlet_patch1 + dof_Lagrangian_dirichlet_patch2));
    Ref<MatrixXd> K_patch1 = K.block(0, 0, dof_patch1, dof_patch1);
    Ref<MatrixXd> K_patch2 = K.block(dof_patch1, dof_patch1, dof_patch2, dof_patch2);
    Ref<MatrixXd> K_Lagrangian_constraints_C0 = K.block(dof_patch1 + dof_patch2, 0, dof_Lagrangian_constaints,
                                                        dof_patch1 + dof_patch2);
    Ref<MatrixXd> Mass_Lagrangian_constraints_C0 = K.block(dof_patch1 + dof_patch2, dof_patch1 + dof_patch2,
                                                           dof_Lagrangian_constaints, dof_Lagrangian_constaints);
    Ref<MatrixXd> K_Lagrangian_constraints_C1 = K.block(dof_patch1 + dof_patch2 + dof_Lagrangian_constaints, 0,
                                                        dof_Lagrangian_constaints,
                                                        dof_patch1 + dof_patch2);
    Ref<MatrixXd> Mass_Lagrangian_constraints_C1 = K.block(dof_patch1 + dof_patch2 + dof_Lagrangian_constaints,
                                                           dof_patch1 + dof_patch2 + dof_Lagrangian_constaints,
                                                           dof_Lagrangian_constaints, dof_Lagrangian_constaints);
    Ref<VectorXd> F_patch1 = F.segment(0, dof_patch1);
    Ref<VectorXd> F_patch2 = F.segment(dof_patch1, dof_patch2);
    for (auto it_Xi = Xi_span_patch1.begin(); it_Xi != Xi_span_patch1.end(); ++it_Xi) {
        double J_Xi = (it_Xi->second - it_Xi->first) / 2;
        double Middle_Xi = (it_Xi->second + it_Xi->first) / 2;
        int i_Xi = Findspan(m_Xi_patch1, p, knots_Xi_patch1, Middle_Xi);
        for (auto it_Eta = Eta_span_patch1.begin(); it_Eta != Eta_span_patch1.end(); ++it_Eta) {
            double J_Eta = (it_Eta->second - it_Eta->first) / 2;
            double Middle_Eta = (it_Eta->second + it_Eta->first) / 2;
            int i_Eta = Findspan(m_Eta_patch1, p, knots_Eta_patch1, Middle_Eta);
            for (int jj_Xi = 0; jj_Xi < gaussian_points; jj_Xi++) {
                double xi = Middle_Xi + J_Xi * gaussian[jj_Xi];
                double **ders_x;
                DersBasisFuns(i_Xi, xi, p, knots_Xi_patch1, 2, ders_x);
                VectorXd Nxi(p + 1), Nxi_xi(p + 1), Nxi_xi_xi(p + 1);
                for (int kk_x = 0; kk_x < p + 1; kk_x++) {
                    Nxi(kk_x) = ders_x[0][kk_x];
                    Nxi_xi(kk_x) = ders_x[1][kk_x];
                    Nxi_xi_xi(kk_x) = ders_x[2][kk_x];
                }
                for (int k = 0; k < 3; k++)
                    delete ders_x[k];
                delete[] ders_x;
                for (int jj_Eta = 0; jj_Eta < gaussian_points; jj_Eta++) {
                    double eta = Middle_Eta + J_Eta * gaussian[jj_Eta];
                    double **ders_y;
                    DersBasisFuns(i_Eta, eta, p, knots_Eta_patch1, 2, ders_y);
                    VectorXd Neta(p + 1), Neta_eta(p + 1), Neta_eta_eta(p + 1);
                    for (int kk_y = 0; kk_y < p + 1; kk_y++) {
                        Neta(kk_y) = ders_y[0][kk_y];
                        Neta_eta(kk_y) = ders_y[1][kk_y];
                        Neta_eta_eta(kk_y) = ders_y[2][kk_y];
                    }
                    for (int k = 0; k < 3; k++)
                        delete ders_y[k];
                    delete[] ders_y;
                    VectorXd Nxi_xiNeta, NxiNeta_eta, Nxi_xi_xiNeta, NxiNeta_eta_eta, Nxi_xiNeta_eta, NxiNeta;
                    Nxi_xiNeta = kroneckerProduct(Nxi_xi, Neta);
                    NxiNeta_eta = kroneckerProduct(Nxi, Neta_eta);
                    Nxi_xi_xiNeta = kroneckerProduct(Nxi_xi_xi, Neta);
                    NxiNeta_eta_eta = kroneckerProduct(Nxi, Neta_eta_eta);
                    Nxi_xiNeta_eta = kroneckerProduct(Nxi_xi, Neta_eta);
                    NxiNeta = kroneckerProduct(Nxi, Neta);
                    double pxpxi, pxpeta, pypxi, pypeta, pxpxi_xi, pxpxi_eta, pxpeta_eta, pypxi_xi, pypxi_eta, pypeta_eta, x, y;
                    Geometry1(xi, eta, pxpxi, pxpeta, pypxi, pypeta, pxpxi_xi, pxpxi_eta, pxpeta_eta, pypxi_xi,
                              pypxi_eta, pypeta_eta, x,
                              y);
                    double force = forceTerm(x, y);
                    double Jacobian = pxpxi * pypeta - pxpeta * pypxi;
                    MatrixXd Hessian(5, 5);
                    Hessian << pxpxi, pypxi, 0, 0, 0, pxpeta, pypeta, 0, 0, 0, pxpxi_xi, pypxi_xi, pxpxi * pxpxi,
                            2 *
                            pxpxi *
                            pypxi,
                            pypxi * pypxi, pxpxi_eta, pypxi_eta, pxpxi * pxpeta, pxpxi * pypeta + pxpeta * pypxi,
                            pypxi * pypeta, pxpeta_eta, pypeta_eta,
                            pxpeta * pxpeta, 2 * pxpeta * pypeta, pypeta * pypeta;
                    MatrixXd Hessian_inv = Hessian.inverse();
                    VectorXd Nx_x_xNy, NxNy_y_y;
                    Nx_x_xNy = Hessian_inv(2, 0) * Nxi_xiNeta + Hessian_inv(2, 1) * NxiNeta_eta + Hessian_inv(2,
                                                                                                              2) *
                                                                                                  Nxi_xi_xiNeta +
                               Hessian_inv(2, 3) * Nxi_xiNeta_eta + Hessian_inv(2, 4) * NxiNeta_eta_eta;
                    NxNy_y_y = Hessian_inv(4, 0) * Nxi_xiNeta + Hessian_inv(4, 1) * NxiNeta_eta + Hessian_inv(4,
                                                                                                              2) *
                                                                                                  Nxi_xi_xiNeta +
                               Hessian_inv(4, 3) * Nxi_xiNeta_eta + Hessian_inv(4, 4) * NxiNeta_eta_eta;
                    for (int kkx = 0; kkx < (p + 1) * (p + 1); kkx++) {
                        for (int kky = 0; kky < (p + 1) * (p + 1); kky++) {
                            double Bx = Nx_x_xNy(kkx) + NxNy_y_y(kkx);
                            double By = Nx_x_xNy(kky) + NxNy_y_y(kky);
                            K_patch1((m_Eta_patch1 - p) * (kkx / (p + 1) + i_Xi - p) + kkx % (p + 1) + i_Eta - p,
                                     (m_Eta_patch1 - p) * (kky / (p + 1) + i_Xi - p) + kky % (p + 1) + i_Eta - p) +=
                                    weight[jj_Xi] * weight[jj_Eta] * Jacobian * Bx * By * J_Xi * J_Eta;
                        }
                        F_patch1((m_Eta_patch1 - p) * (kkx / (p + 1) + i_Xi - p) + kkx % (p + 1) + i_Eta - p) +=
                                weight[jj_Xi] * weight[jj_Eta] * Jacobian * NxiNeta(kkx) * force * J_Xi * J_Eta;
                    }
                }
            }
        }
    }

    for (auto it_Xi = Xi_span_patch2.begin(); it_Xi != Xi_span_patch2.end(); ++it_Xi) {
        double J_Xi = (it_Xi->second - it_Xi->first) / 2;
        double Middle_Xi = (it_Xi->second + it_Xi->first) / 2;
        int i_Xi = Findspan(m_Xi_patch2, p, knots_Xi_patch2, Middle_Xi);
        for (auto it_Eta = Eta_span_patch2.begin(); it_Eta != Eta_span_patch2.end(); ++it_Eta) {
            double J_Eta = (it_Eta->second - it_Eta->first) / 2;
            double Middle_Eta = (it_Eta->second + it_Eta->first) / 2;
            int i_Eta = Findspan(m_Eta_patch2, p, knots_Eta_patch2, Middle_Eta);
            for (int jj_Xi = 0; jj_Xi < gaussian_points; jj_Xi++) {
                double xi = Middle_Xi + J_Xi * gaussian[jj_Xi];
                double **ders_x;
                DersBasisFuns(i_Xi, xi, p, knots_Xi_patch2, 2, ders_x);
                VectorXd Nxi(p + 1), Nxi_xi(p + 1), Nxi_xi_xi(p + 1);
                for (int kk_x = 0; kk_x < p + 1; kk_x++) {
                    Nxi(kk_x) = ders_x[0][kk_x];
                    Nxi_xi(kk_x) = ders_x[1][kk_x];
                    Nxi_xi_xi(kk_x) = ders_x[2][kk_x];
                }
                for (int k = 0; k < 3; k++)
                    delete ders_x[k];
                delete[] ders_x;
                for (int jj_Eta = 0; jj_Eta < gaussian_points; jj_Eta++) {
                    double eta = Middle_Eta + J_Eta * gaussian[jj_Eta];
                    double **ders_y;
                    DersBasisFuns(i_Eta, eta, p, knots_Eta_patch2, 2, ders_y);
                    VectorXd Neta(p + 1), Neta_eta(p + 1), Neta_eta_eta(p + 1);
                    for (int kk_y = 0; kk_y < p + 1; kk_y++) {
                        Neta(kk_y) = ders_y[0][kk_y];
                        Neta_eta(kk_y) = ders_y[1][kk_y];
                        Neta_eta_eta(kk_y) = ders_y[2][kk_y];
                    }
                    for (int k = 0; k < 3; k++)
                        delete ders_y[k];
                    delete[] ders_y;
                    VectorXd Nxi_xiNeta, NxiNeta_eta, Nxi_xi_xiNeta, NxiNeta_eta_eta, Nxi_xiNeta_eta, NxiNeta;
                    Nxi_xiNeta = kroneckerProduct(Nxi_xi, Neta);
                    NxiNeta_eta = kroneckerProduct(Nxi, Neta_eta);
                    Nxi_xi_xiNeta = kroneckerProduct(Nxi_xi_xi, Neta);
                    NxiNeta_eta_eta = kroneckerProduct(Nxi, Neta_eta_eta);
                    Nxi_xiNeta_eta = kroneckerProduct(Nxi_xi, Neta_eta);
                    NxiNeta = kroneckerProduct(Nxi, Neta);
                    double pxpxi, pxpeta, pypxi, pypeta, pxpxi_xi, pxpxi_eta, pxpeta_eta, pypxi_xi, pypxi_eta, pypeta_eta, x, y;
                    Geometry2(xi, eta, pxpxi, pxpeta, pypxi, pypeta, pxpxi_xi, pxpxi_eta, pxpeta_eta, pypxi_xi,
                              pypxi_eta, pypeta_eta, x,
                              y);
                    double force = forceTerm(x, y);
                    double Jacobian = pxpxi * pypeta - pxpeta * pypxi;
                    MatrixXd Hessian(5, 5);
                    Hessian << pxpxi, pypxi, 0, 0, 0, pxpeta, pypeta, 0, 0, 0, pxpxi_xi, pypxi_xi, pxpxi * pxpxi,
                            2 *
                            pxpxi *
                            pypxi,
                            pypxi * pypxi, pxpxi_eta, pypxi_eta, pxpxi * pxpeta, pxpxi * pypeta + pxpeta * pypxi,
                            pypxi * pypeta, pxpeta_eta, pypeta_eta,
                            pxpeta * pxpeta, 2 * pxpeta * pypeta, pypeta * pypeta;
                    MatrixXd Hessian_inv = Hessian.inverse();
                    VectorXd Nx_x_xNy, NxNy_y_y;
                    Nx_x_xNy = Hessian_inv(2, 0) * Nxi_xiNeta + Hessian_inv(2, 1) * NxiNeta_eta + Hessian_inv(2,
                                                                                                              2) *
                                                                                                  Nxi_xi_xiNeta +
                               Hessian_inv(2, 3) * Nxi_xiNeta_eta + Hessian_inv(2, 4) * NxiNeta_eta_eta;
                    NxNy_y_y = Hessian_inv(4, 0) * Nxi_xiNeta + Hessian_inv(4, 1) * NxiNeta_eta + Hessian_inv(4,
                                                                                                              2) *
                                                                                                  Nxi_xi_xiNeta +
                               Hessian_inv(4, 3) * Nxi_xiNeta_eta + Hessian_inv(4, 4) * NxiNeta_eta_eta;
                    for (int kkx = 0; kkx < (p + 1) * (p + 1); kkx++) {
                        for (int kky = 0; kky < (p + 1) * (p + 1); kky++) {
                            double Bx = Nx_x_xNy(kkx) + NxNy_y_y(kkx);
                            double By = Nx_x_xNy(kky) + NxNy_y_y(kky);
                            K_patch2((m_Eta_patch2 - p) * (kkx / (p + 1) + i_Xi - p) + kkx % (p + 1) + i_Eta - p,
                                     (m_Eta_patch2 - p) * (kky / (p + 1) + i_Xi - p) + kky % (p + 1) + i_Eta - p) +=
                                    weight[jj_Xi] * weight[jj_Eta] * Jacobian * Bx * By * J_Xi * J_Eta;
                        }
                        F_patch2((m_Eta_patch2 - p) * (kkx / (p + 1) + i_Xi - p) + kkx % (p + 1) + i_Eta - p) +=
                                weight[jj_Xi] * weight[jj_Eta] * Jacobian * NxiNeta(kkx) * force * J_Xi * J_Eta;
                    }
                }
            }
        }
    }

    for (auto it_Eta = Eta_span_intersection.begin(); it_Eta != Eta_span_intersection.end(); ++it_Eta) {
        double J_Eta = (it_Eta->second - it_Eta->first) / 2;
        double Middle_Eta = (it_Eta->second + it_Eta->first) / 2;
        int i_Xi_patch1 = Findspan(m_Xi_patch1, p, knots_Xi_patch1, 1);
        int i_Xi_patch2 = Findspan(m_Xi_patch2, p, knots_Xi_patch2, 0);
        int i_Eta_patch1 = Findspan(m_Eta_patch1, p, knots_Eta_patch1, Middle_Eta);
        int i_Eta_patch2 = Findspan(m_Eta_patch2, p, knots_Eta_patch2, Middle_Eta);
        int i_Lagrangian_constaints = Findspan(m_Lagrangian_constaints, p, knots_Lagrangian_constaints, Middle_Eta);
        for (int jj_Eta = 0; jj_Eta < gaussian_points; jj_Eta++) {
            double eta = Middle_Eta + J_Eta * gaussian[jj_Eta];
            double **ders_x_patch1, **ders_x_patch2, **ders_y_patch1, **ders_y_patch2, **ders_Lagrangian_constaints;
            DersBasisFuns(i_Xi_patch1, 1, p, knots_Xi_patch1, 1, ders_x_patch1);
            DersBasisFuns(i_Xi_patch2, 0, p, knots_Xi_patch2, 1, ders_x_patch2);
            DersBasisFuns(i_Eta_patch1, eta, p, knots_Eta_patch1, 1, ders_y_patch1);
            DersBasisFuns(i_Eta_patch2, eta, p, knots_Eta_patch2, 1, ders_y_patch2);
            DersBasisFuns(i_Lagrangian_constaints, eta, p, knots_Lagrangian_constaints, 0, ders_Lagrangian_constaints);
            VectorXd Nxi_patch1(p + 1), Neta_patch1(p + 1), Nxi_patch2(p + 1), Neta_patch2(p + 1), Nxi_xi_patch1(
                    p + 1), Neta_eta_patch1(p + 1), Nxi_xi_patch2(p + 1), Neta_eta_patch2(
                    p + 1), Shape_Lagrangian_constaints(dof_Lagrangian_constaints);
            Shape_Lagrangian_constaints.setZero();
            for (int kk_y = 0; kk_y < p + 1; kk_y++) {
                Neta_patch1(kk_y) = ders_y_patch1[0][kk_y];
                Neta_patch2(kk_y) = ders_y_patch2[0][kk_y];
                Neta_eta_patch1(kk_y) = ders_y_patch1[1][kk_y];
                Neta_eta_patch2(kk_y) = ders_y_patch2[1][kk_y];
            }
            for (int kk_y = 0; kk_y < p + 1; kk_y++) {
                Nxi_patch1(kk_y) = ders_x_patch1[0][kk_y];
                Nxi_patch2(kk_y) = ders_x_patch2[0][kk_y];
                Nxi_xi_patch1(kk_y) = ders_x_patch1[1][kk_y];
                Nxi_xi_patch2(kk_y) = ders_x_patch2[1][kk_y];
                Shape_Lagrangian_constaints(i_Lagrangian_constaints - p + kk_y) = ders_Lagrangian_constaints[0][kk_y];
            }
            for (int k = 0; k < 2; k++)
                delete ders_y_patch1[k];
            delete[] ders_y_patch1;
            for (int k = 0; k < 2; k++)
                delete ders_y_patch2[k];
            delete[] ders_y_patch2;
            for (int k = 0; k < 2; k++)
                delete ders_x_patch1[k];
            delete[] ders_x_patch1;
            for (int k = 0; k < 2; k++)
                delete ders_x_patch2[k];
            delete[] ders_x_patch2;
            for (int k = 0; k < 1; k++)
                delete ders_Lagrangian_constaints[k];
            delete[] ders_Lagrangian_constaints;
            double pxpxi_patch1, pxpeta_patch1, pypxi_patch1, pypeta_patch1, x_patch1, y_patch1, pxpxi_patch2, pxpeta_patch2,
                    pypxi_patch2, pypeta_patch2, x_patch2, y_patch2, pxpxi_xi_patch1, pxpxi_eta_patch1, pxpeta_eta_patch1, pypxi_xi_patch1,
                    pypxi_eta_patch1, pypeta_eta_patch1, pxpxi_xi_patch2,
                    pxpxi_eta_patch2, pxpeta_eta_patch2, pypxi_xi_patch2, pypxi_eta_patch2, pypeta_eta_patch2;
            Geometry1(1, eta, pxpxi_patch1, pxpeta_patch1, pypxi_patch1, pypeta_patch1,
                      pxpxi_xi_patch1,
                      pxpxi_eta_patch1, pxpeta_eta_patch1, pypxi_xi_patch1, pypxi_eta_patch1, pypeta_eta_patch1,
                      x_patch1, y_patch1);
            Geometry2(0, eta, pxpxi_patch2, pxpeta_patch2, pypxi_patch2, pypeta_patch2,
                      pxpxi_xi_patch2,
                      pxpxi_eta_patch2, pxpeta_eta_patch2, pypxi_xi_patch2, pypxi_eta_patch2, pypeta_eta_patch2,
                      x_patch2, y_patch2);
            double Jacobian_patch1 = pxpxi_patch1 * pypeta_patch1 - pxpeta_patch1 * pypxi_patch1;
            double Jacobian_patch2 = pxpxi_patch2 * pypeta_patch2 - pxpeta_patch2 * pypxi_patch2;
            double alpha, beta, gamma;
            alpha = pxpxi_patch2 * pypxi_patch1 - pxpxi_patch1 * pypxi_patch2;
            beta = pxpeta_patch1 * pypxi_patch2 - pxpxi_patch2 * pypeta_patch1;
            gamma = pxpeta_patch1 * pypxi_patch1 - pxpxi_patch1 * pypeta_patch1;
            VectorXd NxiNeta_patch1, NxiNeta_patch2, Nxi_xiNeta_patch1, Nxi_xiNeta_patch2, NxiNeta_eta_patch1;
            NxiNeta_patch1 = kroneckerProduct(Nxi_patch1, Neta_patch1);
            NxiNeta_patch2 = kroneckerProduct(Nxi_patch2, Neta_patch2);
            Nxi_xiNeta_patch1 = kroneckerProduct(Nxi_xi_patch1, Neta_patch1);
            Nxi_xiNeta_patch2 = kroneckerProduct(Nxi_xi_patch2, Neta_patch2);
            NxiNeta_eta_patch1 = kroneckerProduct(Nxi_patch1, Neta_eta_patch1);
            Mass_Lagrangian_constraints_C0 -= 1.0 / epsilon * weight[jj_Eta] * Shape_Lagrangian_constaints *
                                              Shape_Lagrangian_constaints.transpose() * J_Eta;
            Mass_Lagrangian_constraints_C1 -= 1.0 / epsilon * weight[jj_Eta] * Shape_Lagrangian_constaints *
                                              Shape_Lagrangian_constaints.transpose() * J_Eta;
            for (int kkx = 0; kkx < (p + 1) * (p + 1); kkx++) {
                K_Lagrangian_constraints_C0.col(
                        ((m_Eta_patch1 - p) * (kkx / (p + 1) + i_Xi_patch1 - p) + kkx % (p + 1) + i_Eta_patch1 - p)) -=
                        weight[jj_Eta] * NxiNeta_patch1(kkx) * Shape_Lagrangian_constaints * J_Eta;
                K_Lagrangian_constraints_C0.col(
                        (dof_patch1 + (m_Eta_patch2 - p) * (kkx / (p + 1) + i_Xi_patch2 - p) + kkx % (p + 1) +
                         i_Eta_patch2 - p)) +=
                        weight[jj_Eta] * NxiNeta_patch2(kkx) * Shape_Lagrangian_constaints * J_Eta;
                K_Lagrangian_constraints_C1.col(
                        ((m_Eta_patch1 - p) * (kkx / (p + 1) + i_Xi_patch1 - p) + kkx % (p + 1) + i_Eta_patch1 - p)) -=
                        weight[jj_Eta] * (alpha * NxiNeta_eta_patch1(kkx) + beta * Nxi_xiNeta_patch1(kkx)) *
                        Shape_Lagrangian_constaints * J_Eta;
                K_Lagrangian_constraints_C1.col(
                        (dof_patch1 + (m_Eta_patch2 - p) * (kkx / (p + 1) + i_Xi_patch2 - p) + kkx % (p + 1) +
                         i_Eta_patch2 - p)) +=
                        weight[jj_Eta] * gamma * Nxi_xiNeta_patch2(kkx) * Shape_Lagrangian_constaints * J_Eta;
            }
        }
    }

    //Boundary problem
    Ref<MatrixXd> K_Lagrangian_patch1_C0 = K.block(dof_patch1 + dof_patch2 + 2 * dof_Lagrangian_constaints, 0,
                                                   dof_Lagrangian_dirichlet_patch1,
                                                   dof_patch1 + dof_patch2);
    Ref<MatrixXd> K_Lagrangian_patch1_C1 = K.block(
            dof_patch1 + dof_patch2 + 2 * dof_Lagrangian_constaints + dof_Lagrangian_dirichlet_patch1, 0,
            dof_Lagrangian_dirichlet_patch1, dof_patch1 + dof_patch2);
    Ref<VectorXd> F_Lagrangian_patch1_C0 = F.segment(dof_patch1 + dof_patch2 + 2 * dof_Lagrangian_constaints,
                                                     dof_Lagrangian_dirichlet_patch1);
    Ref<VectorXd> F_Lagrangian_patch1_C1 = F.segment(
            dof_patch1 + dof_patch2 + 2 * dof_Lagrangian_constaints + dof_Lagrangian_dirichlet_patch1,
            dof_Lagrangian_dirichlet_patch1);

    Ref<MatrixXd> Mass_Lagrangian_patch1_C0 = K.block(dof_patch1 + dof_patch2 + 2 * dof_Lagrangian_constaints,
                                                      dof_patch1 + dof_patch2 + 2 * dof_Lagrangian_constaints,
                                                      dof_Lagrangian_dirichlet_patch1, dof_Lagrangian_dirichlet_patch1);
    Ref<MatrixXd> Mass_Lagrangian_patch1_C1 = K.block(
            dof_patch1 + dof_patch2 + 2 * dof_Lagrangian_constaints + dof_Lagrangian_dirichlet_patch1,
            dof_patch1 + dof_patch2 + 2 * dof_Lagrangian_constaints + dof_Lagrangian_dirichlet_patch1,
            dof_Lagrangian_dirichlet_patch1, dof_Lagrangian_dirichlet_patch1);
    for (auto it_Xi = Xi_span_patch1.begin(); it_Xi != Xi_span_patch1.end(); ++it_Xi) {
        double J_Xi = (it_Xi->second - it_Xi->first) / 2;
        double Middle_Xi = (it_Xi->second + it_Xi->first) / 2;
        int i_Xi = Findspan(m_Xi_patch1, p, knots_Xi_patch1, Middle_Xi);
        int i_Eta = Findspan(m_Eta_patch1, p, knots_Eta_patch2, 1);
        int i_Lagrangian_dirichlet_patch1 = Findspan(m_Lagrangian_dirichlet_patch1, p,
                                                     knots_Lagrangian_dirichlet_patch1, Middle_Xi);
        for (int jj_Xi = 0; jj_Xi < gaussian_points; jj_Xi++) {
            double xi = Middle_Xi + J_Xi * gaussian[jj_Xi];
            double **ders_x, **ders_y, **ders_Lagrangian_dirichlet_patch1;
            DersBasisFuns(i_Xi, xi, p, knots_Xi_patch1, 1, ders_x);
            DersBasisFuns(i_Eta, 1, p, knots_Eta_patch1, 1, ders_y);
            DersBasisFuns(i_Lagrangian_dirichlet_patch1, xi, p, knots_Lagrangian_dirichlet_patch1, 0,
                          ders_Lagrangian_dirichlet_patch1);
            VectorXd Nxi(p + 1), Nxi_xi(p + 1), Neta(p + 1), Neta_eta(p + 1), Shape_Lagrangian_dirichlet_patch1(
                    dof_Lagrangian_dirichlet_patch1);
            Shape_Lagrangian_dirichlet_patch1.setZero();
            for (int kk_x = 0; kk_x < p + 1; kk_x++) {
                Nxi(kk_x) = ders_x[0][kk_x];
                Nxi_xi(kk_x) = ders_x[1][kk_x];
            }
            for (int kk_y = 0; kk_y < p + 1; kk_y++) {
                Neta(kk_y) = ders_y[0][kk_y];
                Neta_eta(kk_y) = ders_y[1][kk_y];
                Shape_Lagrangian_dirichlet_patch1(
                        i_Lagrangian_dirichlet_patch1 - p + kk_y) = ders_Lagrangian_dirichlet_patch1[0][kk_y];
            }
            for (int k = 0; k < 2; k++)
                delete ders_x[k];
            delete[] ders_x;
            for (int k = 0; k < 2; k++)
                delete ders_y[k];
            delete[] ders_y;
            for (int k = 0; k < 1; k++)
                delete ders_Lagrangian_dirichlet_patch1[k];
            delete[] ders_Lagrangian_dirichlet_patch1;

            VectorXd Nxi_xiNeta, NxiNeta_eta, Nxi_xi_xiNeta, NxiNeta_eta_eta, Nxi_xiNeta_eta, NxiNeta;
            Nxi_xiNeta = kroneckerProduct(Nxi_xi, Neta);
            NxiNeta_eta = kroneckerProduct(Nxi, Neta_eta);
            Nxi_xiNeta_eta = kroneckerProduct(Nxi_xi, Neta_eta);
            NxiNeta = kroneckerProduct(Nxi, Neta);
            double pxpxi, pxpeta, pypxi, pypeta, pxpxi_xi, pxpxi_eta, pxpeta_eta, pypxi_xi, pypxi_eta, pypeta_eta, x, y;
            Geometry1(xi, 1, pxpxi, pxpeta, pypxi, pypeta, pxpxi_xi, pxpxi_eta, pxpeta_eta, pypxi_xi,
                      pypxi_eta, pypeta_eta, x,
                      y);
            double Jacobian = pxpxi * pypeta - pxpeta * pypxi;
            VectorXd Nx_xNy, NxNy_y;
            Nx_xNy = 1.0 / Jacobian * (Nxi_xiNeta * pypeta - NxiNeta_eta * pypxi);
            NxNy_y = 1.0 / Jacobian * (-Nxi_xiNeta * pxpeta + NxiNeta_eta * pxpxi);
            Mass_Lagrangian_patch1_C0 -= 1.0 / epsilon * weight[jj_Xi] * Shape_Lagrangian_dirichlet_patch1 *
                                         Shape_Lagrangian_dirichlet_patch1.transpose() * J_Xi;
            Mass_Lagrangian_patch1_C1 -= 1.0 / epsilon * weight[jj_Xi] * Shape_Lagrangian_dirichlet_patch1 *
                                         Shape_Lagrangian_dirichlet_patch1.transpose() * J_Xi;
            for (int kkx = 0; kkx < (p + 1) * (p + 1); kkx++) {
                K_Lagrangian_patch1_C0.col(
                        ((m_Eta_patch1 - p) * (kkx / (p + 1) + i_Xi - p) + kkx % (p + 1) + i_Eta - p)) +=
                        weight[jj_Xi] * NxiNeta(kkx) * Shape_Lagrangian_dirichlet_patch1 * J_Xi * pypxi;
                K_Lagrangian_patch1_C1.col(
                        ((m_Eta_patch1 - p) * (kkx / (p + 1) + i_Xi - p) + kkx % (p + 1) + i_Eta - p)) +=
                        weight[jj_Xi] * Nx_xNy(kkx) * Shape_Lagrangian_dirichlet_patch1 * J_Xi * pypxi;
                F_Lagrangian_patch1_C0 +=
                        weight[jj_Xi] * exactSolution(x, y) * Shape_Lagrangian_dirichlet_patch1 * J_Xi * pypxi;
                F_Lagrangian_patch1_C1 +=
                        weight[jj_Xi] * exactSolution_dx(x, y) * Shape_Lagrangian_dirichlet_patch1 * J_Xi * pypxi;
            }
        }
    }


    Ref<MatrixXd> K_Lagrangian_patch2_C0 = K.block(
            dof_patch1 + dof_patch2 + 2 * (dof_Lagrangian_constaints + dof_Lagrangian_dirichlet_patch1), 0,
            dof_Lagrangian_dirichlet_patch2, dof_patch1 + dof_patch2);
    Ref<MatrixXd> K_Lagrangian_patch2_C1 = K.block(
            dof_patch1 + dof_patch2 + 2 * (dof_Lagrangian_constaints + dof_Lagrangian_dirichlet_patch1) +
            dof_Lagrangian_dirichlet_patch2, 0,
            dof_Lagrangian_dirichlet_patch2, dof_patch1 + dof_patch2);
    Ref<VectorXd> F_Lagrangian_patch2_C0 = F.segment(
            dof_patch1 + dof_patch2 + 2 * (dof_Lagrangian_constaints + dof_Lagrangian_dirichlet_patch1),
            dof_Lagrangian_dirichlet_patch2);
    Ref<VectorXd> F_Lagrangian_patch2_C1 = F.segment(
            dof_patch1 + dof_patch2 + 2 * (dof_Lagrangian_constaints + dof_Lagrangian_dirichlet_patch1) +
            dof_Lagrangian_dirichlet_patch2,
            dof_Lagrangian_dirichlet_patch2);
    Ref<MatrixXd> Mass_Lagrangian_patch2_C0 = K.block(
            dof_patch1 + dof_patch2 + 2 * (dof_Lagrangian_constaints + dof_Lagrangian_dirichlet_patch1),
            dof_patch1 + dof_patch2 + 2 * (dof_Lagrangian_constaints + dof_Lagrangian_dirichlet_patch1),
            dof_Lagrangian_dirichlet_patch2, dof_Lagrangian_dirichlet_patch2);
    Ref<MatrixXd> Mass_Lagrangian_patch2_C1 = K.block(
            dof_patch1 + dof_patch2 + 2 * (dof_Lagrangian_constaints + dof_Lagrangian_dirichlet_patch1) +
            dof_Lagrangian_dirichlet_patch2,
            dof_patch1 + dof_patch2 + 2 * (dof_Lagrangian_constaints + dof_Lagrangian_dirichlet_patch1) +
            dof_Lagrangian_dirichlet_patch2,
            dof_Lagrangian_dirichlet_patch2, dof_Lagrangian_dirichlet_patch2);
    for (auto it_Xi = Xi_span_patch2.begin(); it_Xi != Xi_span_patch2.end(); ++it_Xi) {
        double J_Xi = (it_Xi->second - it_Xi->first) / 2;
        double Middle_Xi = (it_Xi->second + it_Xi->first) / 2;
        int i_Xi = Findspan(m_Xi_patch2, p, knots_Xi_patch2, Middle_Xi);
        int i_Eta = Findspan(m_Eta_patch2, p, knots_Eta_patch2, 1);
        int i_Lagrangian_dirichlet_patch2 = Findspan(m_Lagrangian_dirichlet_patch2, p,
                                                     knots_Lagrangian_dirichlet_patch2, Middle_Xi);
        for (int jj_Xi = 0; jj_Xi < gaussian_points; jj_Xi++) {
            double xi = Middle_Xi + J_Xi * gaussian[jj_Xi];
            double **ders_x, **ders_y, **ders_Lagrangian_dirichlet_patch2;
            DersBasisFuns(i_Xi, xi, p, knots_Xi_patch2, 1, ders_x);
            DersBasisFuns(i_Eta, 1, p, knots_Eta_patch2, 1, ders_y);
            DersBasisFuns(i_Lagrangian_dirichlet_patch2, xi, p, knots_Lagrangian_dirichlet_patch2, 0,
                          ders_Lagrangian_dirichlet_patch2);
            VectorXd Nxi(p + 1), Nxi_xi(p + 1), Neta(p + 1), Neta_eta(p + 1), Shape_Lagrangian_dirichlet_patch2(
                    dof_Lagrangian_dirichlet_patch2);
            Shape_Lagrangian_dirichlet_patch2.setZero();
            for (int kk_x = 0; kk_x < p + 1; kk_x++) {
                Nxi(kk_x) = ders_x[0][kk_x];
                Nxi_xi(kk_x) = ders_x[1][kk_x];
            }
            for (int kk_y = 0; kk_y < p + 1; kk_y++) {
                Neta(kk_y) = ders_y[0][kk_y];
                Neta_eta(kk_y) = ders_y[1][kk_y];
                Shape_Lagrangian_dirichlet_patch2(
                        i_Lagrangian_dirichlet_patch2 - p + kk_y) = ders_Lagrangian_dirichlet_patch2[0][kk_y];
            }
            for (int k = 0; k < 2; k++)
                delete ders_x[k];
            delete[] ders_x;
            for (int k = 0; k < 2; k++)
                delete ders_y[k];
            delete[] ders_y;
            for (int k = 0; k < 1; k++)
                delete ders_Lagrangian_dirichlet_patch2[k];
            delete[] ders_Lagrangian_dirichlet_patch2;

            VectorXd Nxi_xiNeta, NxiNeta_eta, Nxi_xi_xiNeta, NxiNeta_eta_eta, Nxi_xiNeta_eta, NxiNeta;
            Nxi_xiNeta = kroneckerProduct(Nxi_xi, Neta);
            NxiNeta_eta = kroneckerProduct(Nxi, Neta_eta);
            Nxi_xiNeta_eta = kroneckerProduct(Nxi_xi, Neta_eta);
            NxiNeta = kroneckerProduct(Nxi, Neta);
            double pxpxi, pxpeta, pypxi, pypeta, pxpxi_xi, pxpxi_eta, pxpeta_eta, pypxi_xi, pypxi_eta, pypeta_eta, x, y;
            Geometry2(xi, 1, pxpxi, pxpeta, pypxi, pypeta, pxpxi_xi, pxpxi_eta, pxpeta_eta, pypxi_xi,
                      pypxi_eta, pypeta_eta, x,
                      y);
            double Jacobian = pxpxi * pypeta - pxpeta * pypxi;
            VectorXd Nx_xNy, NxNy_y;
            Nx_xNy = 1.0 / Jacobian * (Nxi_xiNeta * pypeta - NxiNeta_eta * pypxi);
            NxNy_y = 1.0 / Jacobian * (-Nxi_xiNeta * pxpeta + NxiNeta_eta * pxpxi);
            Mass_Lagrangian_patch2_C0 -= 1.0 / epsilon * weight[jj_Xi] * Shape_Lagrangian_dirichlet_patch2 *
                                         Shape_Lagrangian_dirichlet_patch2.transpose() * J_Xi;
            Mass_Lagrangian_patch2_C1 -= 1.0 / epsilon * weight[jj_Xi] * Shape_Lagrangian_dirichlet_patch2 *
                                         Shape_Lagrangian_dirichlet_patch2.transpose() * J_Xi;
            for (int kkx = 0; kkx < (p + 1) * (p + 1); kkx++) {
                K_Lagrangian_patch2_C0.col(
                        (dof_patch1 + (m_Eta_patch2 - p) * (kkx / (p + 1) + i_Xi - p) + kkx % (p + 1) + i_Eta - p)) +=
                        weight[jj_Xi] * NxiNeta(kkx) * Shape_Lagrangian_dirichlet_patch2 * J_Xi * pxpxi;
                K_Lagrangian_patch2_C1.col(
                        (dof_patch1 + (m_Eta_patch2 - p) * (kkx / (p + 1) + i_Xi - p) + kkx % (p + 1) + i_Eta - p)) +=
                        weight[jj_Xi] * NxNy_y(kkx) * Shape_Lagrangian_dirichlet_patch2 * J_Xi * pxpxi;
                F_Lagrangian_patch2_C0 +=
                        weight[jj_Xi] * exactSolution(x, y) * Shape_Lagrangian_dirichlet_patch2 * J_Xi * pxpxi;
                F_Lagrangian_patch2_C1 +=
                        weight[jj_Xi] * exactSolution_dy(x, y) * Shape_Lagrangian_dirichlet_patch2 * J_Xi * pxpxi;
            }
        }
    }
    SpMat transform(K.cols() - 2 * (dof_Eta_patch1 + dof_Xi_patch1 + dof_Eta_patch2 + dof_Xi_patch2) + 8, K.cols());
    vector<Eigen::Triplet<double> > coefficients;
    int x_it = 0, y_it = 0;
    for (int i = 0; i < dof_Xi_patch1; i++) {
        for (int j = 0; j < dof_Eta_patch1; j++) {
            if ((i == 0) || (i == 1) || (j == 0) || (j == 1)) {
                y_it++;
            } else {
                coefficients.push_back(Eigen::Triplet<double>(x_it, y_it, 1));
                x_it++;
                y_it++;
            }
        }
    }
    for (int i = 0; i < dof_Xi_patch2; i++) {
        for (int j = 0; j < dof_Eta_patch2; j++) {
            if ((i == 0) || (i == 1) || (j == 0) || (j == 1)) {
                y_it++;
            } else {
                coefficients.push_back(Eigen::Triplet<double>(x_it, y_it, 1));
                x_it++;
                y_it++;
            }
        }
    }
    for (int i = 0;
         i < 2 * (dof_Lagrangian_constaints + dof_Lagrangian_dirichlet_patch1 + dof_Lagrangian_dirichlet_patch2); i++) {
        coefficients.push_back(Eigen::Triplet<double>(x_it, y_it, 1));
        x_it++;
        y_it++;

    }
    transform.setFromTriplets(coefficients.begin(), coefficients.end());
    cout << K << endl;
    K = K.selfadjointView<Lower>();
    cout << K << endl;
    SpMat K_cal = (transform * K * transform.transpose()).sparseView();
    VectorXd F_cal = transform * F;

    return 0;
}

void Geometry1(double xi, double eta, double &pxpxi, double &pxpeta, double &pypxi, double &pypeta,
               double &pxpxi_xi, double &pxpxi_eta, double &pxpeta_eta, double &pypxi_xi, double &pypxi_eta,
               double &pypeta_eta,
               double &x, double &y) {
    double knot_x[] = {0, 0, 1, 1};
    double knot_y[] = {0, 0, 1, 1};
    MatrixXd B_x(2, 2);
    MatrixXd B_y(2, 2);

    B_x << 0, 0, 2, 2;
    B_y << 4, 0, 4, 2;

    int p_x = 1, p_y = 1;
    int m_x = 3, m_y = 3;
    int dof_x = m_x - p_x, dof_y = m_y - p_y;
    int i_x = Findspan(m_x, p_x, knot_x, xi);
    int i_y = Findspan(m_y, p_y, knot_y, eta);
    double **ders_x, **ders_y;
    DersBasisFuns(i_x, xi, p_x, knot_x, 2, ders_x);
    DersBasisFuns(i_y, eta, p_y, knot_y, 2, ders_y);
    SpVec Nxi(dof_x), Nxi_xi(dof_x), Nxi_xi_xi(dof_x), Neta(dof_y), Neta_eta(dof_y), Neta_eta_eta(dof_y);
    for (int kk_x = 0; kk_x < p_x + 1; kk_x++) {
        Nxi.insert(i_x - p_x + kk_x) = ders_x[0][kk_x];
        Nxi_xi.insert(i_x - p_x + kk_x) = ders_x[1][kk_x];
        Nxi_xi_xi.insert(i_x - p_x + kk_x) = ders_x[2][kk_x];
    }
    for (int kk_y = 0; kk_y < p_y + 1; kk_y++) {
        Neta.insert(i_y - p_y + kk_y) = ders_y[0][kk_y];
        Neta_eta.insert(i_y - p_y + kk_y) = ders_y[1][kk_y];
        Neta_eta_eta.insert(i_y - p_y + kk_y) = ders_y[2][kk_y];
    }
    for (int k = 0; k < 3; k++)
        delete ders_x[k];
    delete[] ders_x;
    for (int k = 0; k < 3; k++)
        delete ders_y[k];
    delete[] ders_y;
    MatrixXd pxpxi_temp, pxpeta_temp, pypxi_temp, pypeta_temp, pxpxi_xi_temp, pxpxi_eta_temp, pxpeta_eta_temp,
            pypxi_xi_temp, pypxi_eta_temp, pypeta_eta_temp;
    pxpxi_temp = Neta.transpose() * B_x * Nxi_xi;
    pypxi_temp = Neta.transpose() * B_y * Nxi_xi;
    pxpeta_temp = Neta_eta.transpose() * B_x * Nxi;
    pypeta_temp = Neta_eta.transpose() * B_y * Nxi;
    pxpxi_xi_temp = Neta.transpose() * B_x * Nxi_xi_xi;
    pxpxi_eta_temp = Neta_eta.transpose() * B_x * Nxi_xi;
    pxpeta_eta_temp = Neta_eta_eta.transpose() * B_x * Nxi;
    pypxi_xi_temp = Neta.transpose() * B_y * Nxi_xi_xi;
    pypxi_eta_temp = Neta_eta.transpose() * B_y * Nxi_xi;
    pypeta_eta_temp = Neta_eta_eta.transpose() * B_y * Nxi;
    pxpxi = pxpxi_temp(0, 0);
    pxpeta = pxpeta_temp(0, 0);
    pypxi = pypxi_temp(0, 0);
    pypeta = pypeta_temp(0, 0);
    pxpxi_xi = pxpxi_xi_temp(0, 0);
    pxpxi_eta = pxpxi_eta_temp(0, 0);
    pxpeta_eta = pxpeta_eta_temp(0, 0);
    pypxi_xi = pypxi_xi_temp(0, 0);
    pypxi_eta = pypxi_eta_temp(0, 0);
    pypeta_eta = pypeta_eta_temp(0, 0);
    MatrixXd x0 = Neta.transpose() * B_x * Nxi;
    MatrixXd y0 = Neta.transpose() * B_y * Nxi;
    x = x0(0, 0);
    y = y0(0, 0);
}

void Geometry2(double xi, double eta, double &pxpxi, double &pxpeta, double &pypxi, double &pypeta,
               double &pxpxi_xi, double &pxpxi_eta, double &pxpeta_eta, double &pypxi_xi, double &pypxi_eta,
               double &pypeta_eta,
               double &x, double &y) {
    double knot_x[] = {0, 0, 1, 1};
    double knot_y[] = {0, 0, 1, 1};
    MatrixXd B_x(2, 2);
    MatrixXd B_y(2, 2);


    B_x << 0, 4, 2, 4;
    B_y << 0, 0, 2, 2;

    int p_x = 1, p_y = 1;
    int m_x = 3, m_y = 3;
    int dof_x = m_x - p_x, dof_y = m_y - p_y;
    int i_x = Findspan(m_x, p_x, knot_x, xi);
    int i_y = Findspan(m_y, p_y, knot_y, eta);
    double **ders_x, **ders_y;
    DersBasisFuns(i_x, xi, p_x, knot_x, 2, ders_x);
    DersBasisFuns(i_y, eta, p_y, knot_y, 2, ders_y);
    SpVec Nxi(dof_x), Nxi_xi(dof_x), Nxi_xi_xi(dof_x), Neta(dof_y), Neta_eta(dof_y), Neta_eta_eta(dof_y);
    for (int kk_x = 0; kk_x < p_x + 1; kk_x++) {
        Nxi.insert(i_x - p_x + kk_x) = ders_x[0][kk_x];
        Nxi_xi.insert(i_x - p_x + kk_x) = ders_x[1][kk_x];
        Nxi_xi_xi.insert(i_x - p_x + kk_x) = ders_x[2][kk_x];
    }
    for (int kk_y = 0; kk_y < p_y + 1; kk_y++) {
        Neta.insert(i_y - p_y + kk_y) = ders_y[0][kk_y];
        Neta_eta.insert(i_y - p_y + kk_y) = ders_y[1][kk_y];
        Neta_eta_eta.insert(i_y - p_y + kk_y) = ders_y[2][kk_y];
    }
    for (int k = 0; k < 3; k++)
        delete ders_x[k];
    delete[] ders_x;
    for (int k = 0; k < 3; k++)
        delete ders_y[k];
    delete[] ders_y;
    MatrixXd pxpxi_temp, pxpeta_temp, pypxi_temp, pypeta_temp, pxpxi_xi_temp, pxpxi_eta_temp, pxpeta_eta_temp,
            pypxi_xi_temp, pypxi_eta_temp, pypeta_eta_temp;
    pxpxi_temp = Neta.transpose() * B_x * Nxi_xi;
    pypxi_temp = Neta.transpose() * B_y * Nxi_xi;
    pxpeta_temp = Neta_eta.transpose() * B_x * Nxi;
    pypeta_temp = Neta_eta.transpose() * B_y * Nxi;
    pxpxi_xi_temp = Neta.transpose() * B_x * Nxi_xi_xi;
    pxpxi_eta_temp = Neta_eta.transpose() * B_x * Nxi_xi;
    pxpeta_eta_temp = Neta_eta_eta.transpose() * B_x * Nxi;
    pypxi_xi_temp = Neta.transpose() * B_y * Nxi_xi_xi;
    pypxi_eta_temp = Neta_eta.transpose() * B_y * Nxi_xi;
    pypeta_eta_temp = Neta_eta_eta.transpose() * B_y * Nxi;
    pxpxi = pxpxi_temp(0, 0);
    pxpeta = pxpeta_temp(0, 0);
    pypxi = pypxi_temp(0, 0);
    pypeta = pypeta_temp(0, 0);
    pxpxi_xi = pxpxi_xi_temp(0, 0);
    pxpxi_eta = pxpxi_eta_temp(0, 0);
    pxpeta_eta = pxpeta_eta_temp(0, 0);
    pypxi_xi = pypxi_xi_temp(0, 0);
    pypxi_eta = pypxi_eta_temp(0, 0);
    pypeta_eta = pypeta_eta_temp(0, 0);
    MatrixXd x0 = Neta.transpose() * B_x * Nxi;
    MatrixXd y0 = Neta.transpose() * B_y * Nxi;
    x = x0(0, 0);
    y = y0(0, 0);
}