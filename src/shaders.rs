pub mod denoise_cs           {vulkano_shaders::shader!{ty: "compute",path: "src/shaders/denoise.comp",           include: [],}}
pub mod reproject_cs         {vulkano_shaders::shader!{ty: "compute",path: "src/shaders/reproject.comp",         include: [],}}
pub mod accumulate_cs        {vulkano_shaders::shader!{ty: "compute",path: "src/shaders/accumulate.comp",        include: [],}}
pub mod pre_trace_cs         {vulkano_shaders::shader!{ty: "compute",path: "src/shaders/pre_trace.comp",         include: ["src/shaders/"],}}
pub mod intersect_cs         {vulkano_shaders::shader!{ty: "compute",path: "src/shaders/intersect.comp",         include: ["src/shaders/"],}}
pub mod light_bounce_cs      {vulkano_shaders::shader!{ty: "compute",path: "src/shaders/light_bounce.comp",      include: ["src/shaders/"],}}
pub mod light_occlude_cs     {vulkano_shaders::shader!{ty: "compute",path: "src/shaders/light_occlude.comp",     include: ["src/shaders/"],}}
pub mod light_combine_cs     {vulkano_shaders::shader!{ty: "compute",path: "src/shaders/light_combine.comp",     include: ["src/shaders/"],}}
pub mod normal_blend_cs      {vulkano_shaders::shader!{ty: "compute",path: "src/shaders/normal_blend.comp",      include: ["src/shaders/"],}}
pub mod atrous_cs            {vulkano_shaders::shader!{ty: "compute",path: "src/shaders/atrous.comp",            include: [],}}
pub mod apply_texture_cs     {vulkano_shaders::shader!{ty: "compute",path: "src/shaders/apply_texture.comp",     include: [],}}
pub mod sample_decay_cs      {vulkano_shaders::shader!{ty: "compute",path: "src/shaders/sample_decay.comp",      include: [],}}
pub mod postprocess_cs       {vulkano_shaders::shader!{ty: "compute",path: "src/shaders/postprocess.comp",       include: [],}}
pub mod stratified_sample_cs {vulkano_shaders::shader!{ty: "compute",path: "src/shaders/stratified_sample.comp", include: [],}}
pub mod ray_test_cs          {vulkano_shaders::shader!{ty: "compute",path: "src/shaders/ray_test.comp",          include: ["src/shaders/"],}}
pub mod raycast_cs           {vulkano_shaders::shader!{ty: "compute",path: "src/shaders/raycast.comp",           include: ["src/shaders/"],}}
pub mod path_bounce_cs       {vulkano_shaders::shader!{ty: "compute",path: "src/shaders/path/path_bounce.comp",  include: ["src/shaders/"],}}
pub mod path_light_cs        {vulkano_shaders::shader!{ty: "compute",path: "src/shaders/path/path_light.comp",   include: ["src/shaders/"],}}
pub mod path_occlude_cs      {vulkano_shaders::shader!{ty: "compute",path: "src/shaders/path/path_occlude.comp", include: ["src/shaders/"],}}

// Push Constant Types
pub use denoise_cs::ty::DenoisePushConstantData;
pub use reproject_cs::ty::ReprojectPushConstantData;
pub use intersect_cs::ty::IntersectPushConstants;
pub use pre_trace_cs::ty::PreTracePushConstants;
pub use light_bounce_cs::ty::LightBouncePushConstantData;
pub use light_occlude_cs::ty::LightOccludePushConstantData;
pub use light_combine_cs::ty::LightCombinePushConstantData;
pub use normal_blend_cs::ty::NormalBlendPushConstantData;
pub use atrous_cs::ty::AtrousPushConstantData;
pub use postprocess_cs::ty::PostprocessPushConstantData;
pub use stratified_sample_cs::ty::StratifiedSamplePushConstantData;
pub use sample_decay_cs::ty::SampleDecayPushConstantData;
pub use ray_test_cs::ty::RayTestPushConstants;
pub use raycast_cs::ty::RayCastPushConstantData;
pub use path_bounce_cs::ty::PathBouncePushConstantData;
pub use path_light_cs::ty::PathLightPushConstantData;
pub use path_occlude_cs::ty::PathOccludePushConstantData;


pub struct PushConstantData {
    pub reproject        : ReprojectPushConstantData,
    pub intersect        : IntersectPushConstants,
    pub pre_trace        : PreTracePushConstants,
    pub light_combine    : LightCombinePushConstantData,
    pub atrous           : AtrousPushConstantData,
    pub postprocess      : PostprocessPushConstantData,
    pub stratifiedsample : StratifiedSamplePushConstantData,
    pub sample_decay     : SampleDecayPushConstantData,
    pub raytest          : RayTestPushConstants,
    pub raycast          : RayCastPushConstantData,
    pub path_bounce      : PathBouncePushConstantData,
    pub path_light       : PathLightPushConstantData,
    pub path_occlude     : PathOccludePushConstantData,
}


// Graphics Primitive Types
// pub use light_bounce_cs::ty::BRDF;
pub use light_bounce_cs::ty::PointLight;
pub use light_bounce_cs::ty::DirectionalLight;
pub use light_bounce_cs::ty::SpotLight;
pub use light_combine_cs::ty::Material;

pub use ray_test_cs::ty::RayRequest;
