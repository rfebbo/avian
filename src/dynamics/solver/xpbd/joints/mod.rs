//! XPBD joint constraints.

mod shared;
pub use shared::{FixedAngleConstraintShared, PointConstraintShared};

mod distance;
mod fixed;
mod prismatic;
mod revolute;
#[cfg(feature = "3d")]
mod spherical;

pub use distance::DistanceJointSolverData;
pub use fixed::FixedJointSolverData;
pub use prismatic::PrismaticJointSolverData;
pub use revolute::RevoluteJointSolverData;
#[cfg(feature = "3d")]
pub use spherical::SphericalJointSolverData;

#[cfg(feature = "3d")]
pub(crate) mod velocity_weld;
#[cfg(feature = "3d")]
pub use velocity_weld::{SolveWithContacts, VelocityWeldData};
