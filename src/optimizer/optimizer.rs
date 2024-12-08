use nalgebra::{DMatrix, DVector};

// Gurobi? CPLEX? IPOPT? What is the best solver for the problem?
// Li and Loeliger, 2024?

/// Compute the least square solution of
/// argmin_x ||Cx - y||^2_W
/// where
/// - x is a vector with K entries
/// - y is a vector with N entries
/// - C is a matrix with N rows and K columns
/// - W = inv(V) is a diagonal matrix with non-negative entries
pub fn solve_lstsq(x: &mut DVector<f64>, c: &DMatrix<f64>, y: &DVector<f64>, v: &mut DMatrix<f64>) {
    let mut l = DVector::zeros(c.ncols());
    for (cn, yn) in c.row_iter().zip(y.iter()) {
        // compute the matrix vector product v c_n
        v.mul_to(&cn, &mut l);

        // compute the gain factor
        let g = 1.0 / (yn + cn.dot(&l));

        // compute the residual
        let r = yn - cn.dot(&x);

        // update covariance matrix inplace
        v.syger(-g, &l, &l, 1.0);

        // update the mean vector inplace
        x.axpy(g * r, &l, 1.0);
    }
}

/// other ways to solve least squares (see https://nalgebra.org/docs/user_guide/decompositions_and_lapack/)
/// - direct methods: QR, SVD, Cholesky
/// - iterative methods: gradient descent, conjugate gradient, Newton

/// Update the NUV parameters
pub fn update_nuv(x: &mut DVector<f64>, c: &DMatrix<f64>, y: &DVector<f64>, v: &mut DMatrix<f64>) {
    // each branch has its own update rule
    // TODO
}

/// methods to solve the quadratic problem 
/// - Dual methods
/// - ADMM
pub fn solve_qp() {
    // TODO
}


// | Method                    | Best For                                     | Complexity        | Scalability | Notes                               |
// |---------------------------|----------------------------------------------|-------------------|-------------|-------------------------------------|
// | **Active Set**            | Small problems, few constraints              | \( O( K^3) \)     | Moderate    | Iterative; fewer iterations for small \( m \). |
// | **Interior Point**        | Large, sparse, or dense problems             | \( O( K^3) \)     | High        | Requires \( Q \) positive definite. |
// | **Gradient Projection**   | Simple constraints, large-scale problems     | \( O(N K^2) \)    | High        | Efficient for box-constrained QP.  |
// | **Dual Methods**          | Many constraints, fewer variables            | \( O(NK) \)       | Moderate    | Suitable for structured problems.  |
// | **ADMM**                  | Large-scale, separable problems              | \( O(NK) \)       | Very High   | Parallelizable.                     |
// | **Direct Solvers (KKT)**  | Dense \( Q \), small-to-medium problems      | \( O( K^3) \)     | Low         | Best for dense, small \( n \).     |

/// Determine the solution of the inequality constrained min-norm problem using the dual forward filtering backward deciding algorithm
/// See Li and Loeliger, 2025
pub fn dfdbd(x: &mut DVector<f64>, c: &DMatrix<f64>, y: &DVector<f64>, v: &mut DMatrix<f64>) {
    let mut l = DVector::zeros(c.ncols());
    for (cn, yn) in c.row_iter().zip(y.iter()) {
        // forward filtering

        // compute the matrix vector product v c_n
        v.mul_to(&cn, &mut l);

        // compute the gain factor
        let g = 1.0 / (yn + cn.dot(&l));

        // compute the residual
        let r = yn - cn.dot(&x);

        // update covariance matrix inplace
        v.syger(-g, &l, &l, 1.0);

        // update the mean vector inplace
        x.axpy(g * r, &l, 1.0);

        // backward deciding
        
    }
}