use rand::distributions::Uniform;
use rusty_snn::snn::network::Network;

fn main() {
    println!("Hello, world!");

    let network = Network::new_random_fin(1000,1000, Uniform::from(0.0..1.0), Uniform::from(0.0..1.0), Uniform::from(0..10), Uniform::from(0.0..1.0));
    network.save_to("data/network.json").unwrap();
    let network2 = Network::load_from("data/network.json").unwrap();
}
