use rusty_snn::spike_train::sampler::PeriodicSpikeTrainSampler;
use rusty_snn::spike_train::spike_train::PeriodicSpikeTrain;

use argmin::core::{CostFunction, Executor, State};
use argmin::solver::brent::BrentOpt;
use argmin::solver::simulatedannealing::SimulatedAnnealing;
use itertools::Itertools;
use itertools::enumerate;

use plotters::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

struct Precision {
    ref_firing_times: Vec<Vec<f64>>,
    sim_firing_times: Vec<Vec<f64>>,
    period: f64,
    num_channels: usize,
}

impl CostFunction for Precision {
    type Param = f64;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let shift = *param;
        let mut value = 0.0;
        for (ref_times, sim_times) in self
            .ref_firing_times
            .iter()
            .zip_eq(self.sim_firing_times.iter())
        {
            value = match (ref_times.is_empty(), sim_times.is_empty()) {
                (true, true) => 0.0,
                (true, false) | (false, true) => 1.0,
                (false, false) => {
                    let tmp = ref_times
                        .iter()
                        .cartesian_product(sim_times.iter())
                        .map(|(ref_t, sim_t)| {
                            let mod_dist = ((ref_t - sim_t - shift).rem_euclid(self.period))
                                .min((sim_t + shift - ref_t).rem_euclid(self.period));
                            if mod_dist < 0.5 {
                                2.0 * mod_dist - 1.0
                            } else {
                                0.0
                            }
                        })
                        .sum::<f64>();
                    tmp / ref_times.len() as f64
                }
            }
        }
        Ok((value / self.num_channels as f64))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Hello, world!");
    
    // let mut rng = StdRng::seed_from_u64(42);

    let period = 50.0;
    let firing_rate = 0.2;
    let num_channels = 100;

    let mut rng = StdRng::seed_from_u64(42);

    let sampler = PeriodicSpikeTrainSampler::new(period, firing_rate).unwrap();

    let ref_firing_times = sampler.sample(num_channels, &mut rng);
    let sim_firing_times = sampler.sample(num_channels, &mut rng);

    // let ref_firing_times = vec![vec![1.0, 3.0, 2.0], vec![2.0, 3.1], vec![3.9, 1.1], vec![1.1, 2.7, 3.9], vec![2.8, 3.9]];
    // let sim_firing_times = vec![vec![2.0, 3.0, 4.0], vec![1.2, 3.2], vec![3.9, 0.1], vec![0.9, 3.1], vec![0.1,1.2]];
    // let num_channels = 5;

    // let ref_firing_times = vec![vec![1.0, 3.0, 2.0], vec![2.0, 3.1], vec![3.9, 1.1], vec![1.1, 2.7, 3.9], vec![2.8, 3.9]];
    // let sim_firing_times = vec![vec![2.0, 3.0, 4.0], vec![1.2, 3.2], vec![3.9, 0.1], vec![0.9, 3.1], vec![0.1,1.2]];
    // let num_channels = 5;

    // let ref_firing_times = vec![vec![1.0, 3.0, 2.0], vec![2.0, 3.1], vec![3.9, 1.1]];
    // let sim_firing_times = vec![vec![2.0, 3.0], vec![1.2, 3.2], vec![3.9, 0.1]];
    // let num_channels = 3;

    // let ref_firing_times = vec![vec![2.0, 3.1], vec![3.9, 1.1]];
    // let sim_firing_times = vec![vec![2.0, 3.0], vec![3.9, 0.1]];
    // let num_channels = 2;

    // let ref_firing_times = vec![vec![1.0, 3.0, 2.0]];
    // let sim_firing_times = vec![vec![2.0, 3.0]];
    // let num_channels = 1;

    let cost = Precision {
        ref_firing_times,
        sim_firing_times,
        period,
        num_channels,
    };

    // Create a drawing area
    let root_area = BitMapBackend::new("plot.png", (800, 600)).into_drawing_area();
    root_area.fill(&WHITE)?;

    // Create a chart builder
    let mut chart = ChartBuilder::on(&root_area)
        .caption("Precision", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.0..period, 0.0..0.1)?;

    // Configure the mesh and labels
    chart.configure_mesh().draw()?;

    // Plot the function
    chart.draw_series(LineSeries::new(
        (0..1000).map(|x| x as f64 * period / 1000.).map(|x| (x, - cost.cost(&x).unwrap())),
        &RED,
    ))?;

    // Save the plot
    root_area.present()?;
    
    Ok(())

    // println!("Cost at 0.0: {:?}", cost.cost(&0.0));
    // println!("Cost at 1.0: {:?}", cost.cost(&1.0));

    // // let solver = BrentOpt::new(-period/2.,period/2.);
    // let solver = SimulatedAnnealing::new(1.0);

    // let res = Executor::new(cost, solver).run().unwrap();
    // println!("Best shift: {:?}", res.state().get_best_param().unwrap());
    // println!("Best cost: {:?}", res.state().get_best_cost());
    // println!("Precision: {:?}", - res.state().get_best_cost());
    // println!("Status: {:?}", res.state().get_termination_status());
    // println!("Termination reason: {:?}", res.state().get_termination_reason());
}
