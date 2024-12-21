(function() {var implementors = {
"mysten_common":[["impl&lt;'a, K: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/cmp/trait.Eq.html\" title=\"trait core::cmp::Eq\">Eq</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/hash/trait.Hash.html\" title=\"trait core::hash::Hash\">Hash</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/marker/trait.Unpin.html\" title=\"trait core::marker::Unpin\">Unpin</a>, V: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/marker/trait.Unpin.html\" title=\"trait core::marker::Unpin\">Unpin</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/future/future/trait.Future.html\" title=\"trait core::future::future::Future\">Future</a> for <a class=\"struct\" href=\"mysten_common/sync/notify_read/struct.Registration.html\" title=\"struct mysten_common::sync::notify_read::Registration\">Registration</a>&lt;'a, K, V&gt;"]],
"mysten_metrics":[["impl&lt;'a, F: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/future/future/trait.Future.html\" title=\"trait core::future::future::Future\">Future</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/future/future/trait.Future.html\" title=\"trait core::future::future::Future\">Future</a> for <a class=\"struct\" href=\"mysten_metrics/struct.GaugeGuardFuture.html\" title=\"struct mysten_metrics::GaugeGuardFuture\">GaugeGuardFuture</a>&lt;'a, F&gt;"],["impl&lt;F&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/future/future/trait.Future.html\" title=\"trait core::future::future::Future\">Future</a> for <a class=\"struct\" href=\"mysten_metrics/struct.CancelMonitor.html\" title=\"struct mysten_metrics::CancelMonitor\">CancelMonitor</a>&lt;F&gt;<div class=\"where\">where\n    F: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/future/future/trait.Future.html\" title=\"trait core::future::future::Future\">Future</a>,</div>"],["impl&lt;F: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/future/future/trait.Future.html\" title=\"trait core::future::future::Future\">Future</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/future/future/trait.Future.html\" title=\"trait core::future::future::Future\">Future</a> for <a class=\"struct\" href=\"mysten_metrics/struct.MonitoredScopeFuture.html\" title=\"struct mysten_metrics::MonitoredScopeFuture\">MonitoredScopeFuture</a>&lt;F&gt;"]],
"mysten_network":[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/future/future/trait.Future.html\" title=\"trait core::future::future::Future\">Future</a> for <a class=\"struct\" href=\"mysten_network/client/struct.CachingFuture.html\" title=\"struct mysten_network::client::CachingFuture\">CachingFuture</a>"],["impl&lt;Fut, B, E, ResponseHandlerT&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/future/future/trait.Future.html\" title=\"trait core::future::future::Future\">Future</a> for <a class=\"struct\" href=\"mysten_network/callback/struct.ResponseFuture.html\" title=\"struct mysten_network::callback::ResponseFuture\">ResponseFuture</a>&lt;Fut, ResponseHandlerT&gt;<div class=\"where\">where\n    Fut: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/future/future/trait.Future.html\" title=\"trait core::future::future::Future\">Future</a>&lt;Output = <a class=\"enum\" href=\"https://doc.rust-lang.org/1.81.0/core/result/enum.Result.html\" title=\"enum core::result::Result\">Result</a>&lt;Response&lt;B&gt;, E&gt;&gt;,\n    B: Body&lt;Error: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/fmt/trait.Display.html\" title=\"trait core::fmt::Display\">Display</a> + 'static&gt;,\n    E: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/fmt/trait.Display.html\" title=\"trait core::fmt::Display\">Display</a> + 'static,\n    ResponseHandlerT: <a class=\"trait\" href=\"mysten_network/callback/trait.ResponseHandler.html\" title=\"trait mysten_network::callback::ResponseHandler\">ResponseHandler</a>,</div>"]],
"sui_faucet":[["impl&lt;Res&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/future/future/trait.Future.html\" title=\"trait core::future::future::Future\">Future</a> for <a class=\"struct\" href=\"sui_faucet/metrics_layer/struct.RequestMetricsFuture.html\" title=\"struct sui_faucet::metrics_layer::RequestMetricsFuture\">RequestMetricsFuture</a>&lt;Res&gt;"]]
};if (window.register_implementors) {window.register_implementors(implementors);} else {window.pending_implementors = implementors;}})()