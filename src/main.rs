use rusty_snn::snn::network::Network;

fn main() {
    println!("Hello, world!");

    let snn = Network::new_random_fin(10, 10).unwrap();
}
