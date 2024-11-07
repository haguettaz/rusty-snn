use tokio::sync::broadcast;
use tokio::task;
use tokio::time::{sleep, Duration};
use std::sync::Arc;

#[derive(Debug, Clone)]
struct Message {
    source: usize,
    value: usize,
}

#[tokio::main]
async fn main() {
    // Create a broadcast channel with a buffer size of 16
    let (tx, _) = broadcast::channel(16);

    // Create a barrier to synchronize the units
    let barrier = Arc::new(tokio::sync::Barrier::new(3));

    // Spawn computational units
    let mut handles = vec![];
    for i in [2,3,5].iter() {
        let tx = tx.clone();
        let mut rx = tx.subscribe();
        let barrier = barrier.clone();

        let handle = task::spawn(async move {
            let mut result = 0;
            loop {
                // Check for new messages
                match rx.try_recv() {
                    Ok(msg) => println!("Unit {} received new message: {:?}", i, msg),
                    Err(broadcast::error::TryRecvError::Empty) => {
                        // No message available, continue with computation
                        println!("Unit {} is doing computation", i);
                        sleep(Duration::from_millis(100)).await;

                        // Perform some computation
                        result += 1; // Example computation

                        // If computation yields a specific value, send a message
                        if result % i == 0 {
                            println!("Unit {} sends new message from result {}", i, result);
                            tx.send(Message {source:*i, value:result}).unwrap();
                        }
                    }
                    Err(broadcast::error::TryRecvError::Closed) => break,
                    Err(broadcast::error::TryRecvError::Lagged(_)) => {
                        println!("Unit {} lagged", i);
                    }
                }

                // Synchronize with other units
                barrier.wait().await;
            }
        });

        handles.push(handle);
    }

    // Wait for all tasks to complete (and launch them)
    for handle in handles {
        handle.await.unwrap();
    }
}