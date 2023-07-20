use std::collections::HashMap;
use async_trait::async_trait;

use tokio::sync::mpsc;
use tokio::time::{sleep, Duration};

pub type UniqueId = u16;

#[derive(Debug)]
pub struct NetworkMessage {
    pub src: UniqueId,
    pub dst: UniqueId,
    pub payload: String,
}

#[async_trait]
pub trait Agent {
    fn new(
        id: UniqueId, 
        in_channel: mpsc::Receiver<NetworkMessage>, 
        out_channel: mpsc::Sender<NetworkMessage>,
        attrs: HashMap<String, String>) 
    -> Self;

    async fn run(&mut self);
}


/*****************************************************************************************
 *                                        Echo Agent                                     *
 *****************************************************************************************/
pub struct EchoAgent {
    id: UniqueId,
    in_channel: mpsc::Receiver<NetworkMessage>,
}

#[async_trait]
impl Agent for EchoAgent {

    fn new(id: UniqueId,
        in_channel: mpsc::Receiver<NetworkMessage>, 
        _out_channel: mpsc::Sender<NetworkMessage>, 
        _attrs: HashMap<String, String>) 
    -> Self {
        EchoAgent {
            id, 
            in_channel,
        }
    }

    async fn run(&mut self) {
        println!("Starting Echo agent {}", self.id);
        while let Some(msg) = self.in_channel.recv().await {
            assert!(msg.dst == self.id);
            println!("Echo agent received from agent {}:\n\t{}", msg.src, msg.payload);
        }
    }
}

/*****************************************************************************************
 *                                        Ping Agent                                     *
 *****************************************************************************************/

pub struct PingAgent {
    id: UniqueId,
    out_channel: mpsc::Sender<NetworkMessage>,
    target: UniqueId,
}

#[async_trait]
impl Agent for PingAgent {

    fn new(id: UniqueId,
        _in_channel: mpsc::Receiver<NetworkMessage>, 
        out_channel: mpsc::Sender<NetworkMessage>, 
        attrs: HashMap<String, String>) 
    -> Self {
        PingAgent {
            id, 
            out_channel,
            target: attrs["target"].trim().parse().unwrap(),
        }
    }

    async fn run(&mut self) {
        println!("Starting Ping agent {}", self.id);
        let mut count = 0;
        loop {
            let out = NetworkMessage { 
                src: self.id,  // TODO: setting src should be automated 
                dst: self.target, 
                payload: format!("Hello #{} from Ping agent {}", count, self.id),  
            };

            self.out_channel.send(out).await.expect("Send failed");
            sleep(Duration::from_millis(1_000)).await;
            count += 1
        }
    }
}