(function() {var type_impls = {
"sui_types":[["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Clone-for-PrivateKey%3CG%3E\" class=\"impl\"><a href=\"#impl-Clone-for-PrivateKey%3CG%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;G&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a> for PrivateKey&lt;G&gt;<div class=\"where\">where\n    G: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a> + GroupElement,\n    &lt;G as GroupElement&gt;::ScalarType: Zeroize + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a>,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.clone\" class=\"method trait-impl\"><a href=\"#method.clone\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/1.81.0/core/clone/trait.Clone.html#tymethod.clone\" class=\"fn\">clone</a>(&amp;self) -&gt; PrivateKey&lt;G&gt;</h4></section></summary><div class='docblock'>Returns a copy of the value. <a href=\"https://doc.rust-lang.org/1.81.0/core/clone/trait.Clone.html#tymethod.clone\">Read more</a></div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.clone_from\" class=\"method trait-impl\"><span class=\"rightside\"><span class=\"since\" title=\"Stable since Rust version 1.0.0\">1.0.0</span> · <a class=\"src\" href=\"https://doc.rust-lang.org/1.81.0/src/core/clone.rs.html#172\">source</a></span><a href=\"#method.clone_from\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/1.81.0/core/clone/trait.Clone.html#method.clone_from\" class=\"fn\">clone_from</a>(&amp;mut self, source: <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.81.0/std/primitive.reference.html\">&amp;Self</a>)</h4></section></summary><div class='docblock'>Performs copy-assignment from <code>source</code>. <a href=\"https://doc.rust-lang.org/1.81.0/core/clone/trait.Clone.html#method.clone_from\">Read more</a></div></details></div></details>","Clone","sui_types::crypto::RandomnessPrivateKey"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Debug-for-PrivateKey%3CG%3E\" class=\"impl\"><a href=\"#impl-Debug-for-PrivateKey%3CG%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;G&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/fmt/trait.Debug.html\" title=\"trait core::fmt::Debug\">Debug</a> for PrivateKey&lt;G&gt;<div class=\"where\">where\n    G: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/fmt/trait.Debug.html\" title=\"trait core::fmt::Debug\">Debug</a> + GroupElement,\n    &lt;G as GroupElement&gt;::ScalarType: Zeroize + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/fmt/trait.Debug.html\" title=\"trait core::fmt::Debug\">Debug</a>,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.fmt\" class=\"method trait-impl\"><a href=\"#method.fmt\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/1.81.0/core/fmt/trait.Debug.html#tymethod.fmt\" class=\"fn\">fmt</a>(&amp;self, f: &amp;mut <a class=\"struct\" href=\"https://doc.rust-lang.org/1.81.0/core/fmt/struct.Formatter.html\" title=\"struct core::fmt::Formatter\">Formatter</a>&lt;'_&gt;) -&gt; <a class=\"enum\" href=\"https://doc.rust-lang.org/1.81.0/core/result/enum.Result.html\" title=\"enum core::result::Result\">Result</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.81.0/std/primitive.unit.html\">()</a>, <a class=\"struct\" href=\"https://doc.rust-lang.org/1.81.0/core/fmt/struct.Error.html\" title=\"struct core::fmt::Error\">Error</a>&gt;</h4></section></summary><div class='docblock'>Formats the value using the given formatter. <a href=\"https://doc.rust-lang.org/1.81.0/core/fmt/trait.Debug.html#tymethod.fmt\">Read more</a></div></details></div></details>","Debug","sui_types::crypto::RandomnessPrivateKey"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Deserialize%3C'de%3E-for-PrivateKey%3CG%3E\" class=\"impl\"><a href=\"#impl-Deserialize%3C'de%3E-for-PrivateKey%3CG%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;'de, G&gt; <a class=\"trait\" href=\"https://docs.rs/serde/1.0.210/serde/de/trait.Deserialize.html\" title=\"trait serde::de::Deserialize\">Deserialize</a>&lt;'de&gt; for PrivateKey&lt;G&gt;<div class=\"where\">where\n    G: GroupElement,\n    &lt;G as GroupElement&gt;::ScalarType: Zeroize + <a class=\"trait\" href=\"https://docs.rs/serde/1.0.210/serde/de/trait.Deserialize.html\" title=\"trait serde::de::Deserialize\">Deserialize</a>&lt;'de&gt;,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.deserialize\" class=\"method trait-impl\"><a href=\"#method.deserialize\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://docs.rs/serde/1.0.210/serde/de/trait.Deserialize.html#tymethod.deserialize\" class=\"fn\">deserialize</a>&lt;__D&gt;(\n    __deserializer: __D,\n) -&gt; <a class=\"enum\" href=\"https://doc.rust-lang.org/1.81.0/core/result/enum.Result.html\" title=\"enum core::result::Result\">Result</a>&lt;PrivateKey&lt;G&gt;, &lt;__D as <a class=\"trait\" href=\"https://docs.rs/serde/1.0.210/serde/de/trait.Deserializer.html\" title=\"trait serde::de::Deserializer\">Deserializer</a>&lt;'de&gt;&gt;::<a class=\"associatedtype\" href=\"https://docs.rs/serde/1.0.210/serde/de/trait.Deserializer.html#associatedtype.Error\" title=\"type serde::de::Deserializer::Error\">Error</a>&gt;<div class=\"where\">where\n    __D: <a class=\"trait\" href=\"https://docs.rs/serde/1.0.210/serde/de/trait.Deserializer.html\" title=\"trait serde::de::Deserializer\">Deserializer</a>&lt;'de&gt;,</div></h4></section></summary><div class='docblock'>Deserialize this value from the given Serde deserializer. <a href=\"https://docs.rs/serde/1.0.210/serde/de/trait.Deserialize.html#tymethod.deserialize\">Read more</a></div></details></div></details>","Deserialize<'de>","sui_types::crypto::RandomnessPrivateKey"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Drop-for-PrivateKey%3CG%3E\" class=\"impl\"><a href=\"#impl-Drop-for-PrivateKey%3CG%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;G&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/ops/drop/trait.Drop.html\" title=\"trait core::ops::drop::Drop\">Drop</a> for PrivateKey&lt;G&gt;<div class=\"where\">where\n    G: GroupElement,\n    &lt;G as GroupElement&gt;::ScalarType: Zeroize,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.drop\" class=\"method trait-impl\"><a href=\"#method.drop\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/1.81.0/core/ops/drop/trait.Drop.html#tymethod.drop\" class=\"fn\">drop</a>(&amp;mut self)</h4></section></summary><div class='docblock'>Executes the destructor for this type. <a href=\"https://doc.rust-lang.org/1.81.0/core/ops/drop/trait.Drop.html#tymethod.drop\">Read more</a></div></details></div></details>","Drop","sui_types::crypto::RandomnessPrivateKey"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-PartialEq-for-PrivateKey%3CG%3E\" class=\"impl\"><a href=\"#impl-PartialEq-for-PrivateKey%3CG%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;G&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/cmp/trait.PartialEq.html\" title=\"trait core::cmp::PartialEq\">PartialEq</a> for PrivateKey&lt;G&gt;<div class=\"where\">where\n    G: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/cmp/trait.PartialEq.html\" title=\"trait core::cmp::PartialEq\">PartialEq</a> + GroupElement,\n    &lt;G as GroupElement&gt;::ScalarType: Zeroize + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/cmp/trait.PartialEq.html\" title=\"trait core::cmp::PartialEq\">PartialEq</a>,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.eq\" class=\"method trait-impl\"><a href=\"#method.eq\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/1.81.0/core/cmp/trait.PartialEq.html#tymethod.eq\" class=\"fn\">eq</a>(&amp;self, other: &amp;PrivateKey&lt;G&gt;) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.81.0/std/primitive.bool.html\">bool</a></h4></section></summary><div class='docblock'>This method tests for <code>self</code> and <code>other</code> values to be equal, and is used\nby <code>==</code>.</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.ne\" class=\"method trait-impl\"><span class=\"rightside\"><span class=\"since\" title=\"Stable since Rust version 1.0.0\">1.0.0</span> · <a class=\"src\" href=\"https://doc.rust-lang.org/1.81.0/src/core/cmp.rs.html#262\">source</a></span><a href=\"#method.ne\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/1.81.0/core/cmp/trait.PartialEq.html#method.ne\" class=\"fn\">ne</a>(&amp;self, other: <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.81.0/std/primitive.reference.html\">&amp;Rhs</a>) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.81.0/std/primitive.bool.html\">bool</a></h4></section></summary><div class='docblock'>This method tests for <code>!=</code>. The default implementation is almost always\nsufficient, and should not be overridden without very good reason.</div></details></div></details>","PartialEq","sui_types::crypto::RandomnessPrivateKey"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-PrivateKey%3CG%3E\" class=\"impl\"><a href=\"#impl-PrivateKey%3CG%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;G&gt; PrivateKey&lt;G&gt;<div class=\"where\">where\n    G: GroupElement + <a class=\"trait\" href=\"https://docs.rs/serde/1.0.210/serde/ser/trait.Serialize.html\" title=\"trait serde::ser::Serialize\">Serialize</a>,\n    &lt;G as GroupElement&gt;::ScalarType: FiatShamirChallenge + Zeroize,</div></h3></section></summary><div class=\"impl-items\"><section id=\"method.new\" class=\"method\"><h4 class=\"code-header\">pub fn <a class=\"fn\">new</a>&lt;R&gt;(rng: <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.81.0/std/primitive.reference.html\">&amp;mut R</a>) -&gt; PrivateKey&lt;G&gt;<div class=\"where\">where\n    R: AllowedRng,</div></h4></section><section id=\"method.from\" class=\"method\"><h4 class=\"code-header\">pub fn <a class=\"fn\">from</a>(sc: &lt;G as GroupElement&gt;::ScalarType) -&gt; PrivateKey&lt;G&gt;</h4></section></div></details>",0,"sui_types::crypto::RandomnessPrivateKey"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Serialize-for-PrivateKey%3CG%3E\" class=\"impl\"><a href=\"#impl-Serialize-for-PrivateKey%3CG%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;G&gt; <a class=\"trait\" href=\"https://docs.rs/serde/1.0.210/serde/ser/trait.Serialize.html\" title=\"trait serde::ser::Serialize\">Serialize</a> for PrivateKey&lt;G&gt;<div class=\"where\">where\n    G: GroupElement,\n    &lt;G as GroupElement&gt;::ScalarType: Zeroize + <a class=\"trait\" href=\"https://docs.rs/serde/1.0.210/serde/ser/trait.Serialize.html\" title=\"trait serde::ser::Serialize\">Serialize</a>,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.serialize\" class=\"method trait-impl\"><a href=\"#method.serialize\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://docs.rs/serde/1.0.210/serde/ser/trait.Serialize.html#tymethod.serialize\" class=\"fn\">serialize</a>&lt;__S&gt;(\n    &amp;self,\n    __serializer: __S,\n) -&gt; <a class=\"enum\" href=\"https://doc.rust-lang.org/1.81.0/core/result/enum.Result.html\" title=\"enum core::result::Result\">Result</a>&lt;&lt;__S as <a class=\"trait\" href=\"https://docs.rs/serde/1.0.210/serde/ser/trait.Serializer.html\" title=\"trait serde::ser::Serializer\">Serializer</a>&gt;::<a class=\"associatedtype\" href=\"https://docs.rs/serde/1.0.210/serde/ser/trait.Serializer.html#associatedtype.Ok\" title=\"type serde::ser::Serializer::Ok\">Ok</a>, &lt;__S as <a class=\"trait\" href=\"https://docs.rs/serde/1.0.210/serde/ser/trait.Serializer.html\" title=\"trait serde::ser::Serializer\">Serializer</a>&gt;::<a class=\"associatedtype\" href=\"https://docs.rs/serde/1.0.210/serde/ser/trait.Serializer.html#associatedtype.Error\" title=\"type serde::ser::Serializer::Error\">Error</a>&gt;<div class=\"where\">where\n    __S: <a class=\"trait\" href=\"https://docs.rs/serde/1.0.210/serde/ser/trait.Serializer.html\" title=\"trait serde::ser::Serializer\">Serializer</a>,</div></h4></section></summary><div class='docblock'>Serialize this value into the given Serde serializer. <a href=\"https://docs.rs/serde/1.0.210/serde/ser/trait.Serialize.html#tymethod.serialize\">Read more</a></div></details></div></details>","Serialize","sui_types::crypto::RandomnessPrivateKey"],["<section id=\"impl-Eq-for-PrivateKey%3CG%3E\" class=\"impl\"><a href=\"#impl-Eq-for-PrivateKey%3CG%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;G&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/cmp/trait.Eq.html\" title=\"trait core::cmp::Eq\">Eq</a> for PrivateKey&lt;G&gt;<div class=\"where\">where\n    G: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/cmp/trait.Eq.html\" title=\"trait core::cmp::Eq\">Eq</a> + GroupElement,\n    &lt;G as GroupElement&gt;::ScalarType: Zeroize + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/cmp/trait.Eq.html\" title=\"trait core::cmp::Eq\">Eq</a>,</div></h3></section>","Eq","sui_types::crypto::RandomnessPrivateKey"],["<section id=\"impl-StructuralPartialEq-for-PrivateKey%3CG%3E\" class=\"impl\"><a href=\"#impl-StructuralPartialEq-for-PrivateKey%3CG%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;G&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.81.0/core/marker/trait.StructuralPartialEq.html\" title=\"trait core::marker::StructuralPartialEq\">StructuralPartialEq</a> for PrivateKey&lt;G&gt;<div class=\"where\">where\n    G: GroupElement,\n    &lt;G as GroupElement&gt;::ScalarType: Zeroize,</div></h3></section>","StructuralPartialEq","sui_types::crypto::RandomnessPrivateKey"],["<section id=\"impl-ZeroizeOnDrop-for-PrivateKey%3CG%3E\" class=\"impl\"><a href=\"#impl-ZeroizeOnDrop-for-PrivateKey%3CG%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;G&gt; ZeroizeOnDrop for PrivateKey&lt;G&gt;<div class=\"where\">where\n    G: GroupElement,\n    &lt;G as GroupElement&gt;::ScalarType: Zeroize,</div></h3></section>","ZeroizeOnDrop","sui_types::crypto::RandomnessPrivateKey"]]
};if (window.register_type_impls) {window.register_type_impls(type_impls);} else {window.pending_type_impls = type_impls;}})()