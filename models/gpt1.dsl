# === FORWARD PASS ===
embedding t0, wte, tok
embedding t1, wpe, pos
add t2, t0, t1
layernorm t3, t2, h0_ln1_g, h0_ln1_b
matmul t4, t3, h0_attn_wq_w
add t5, t4, h0_attn_wq_b
reshape t6, t5, 1,512,12,64
transpose t7, t6, 1, 2
reshape t8, t7, 12,512,64
matmul t9, t3, h0_attn_wk_w
add t10, t9, h0_attn_wk_b
reshape t11, t10, 1,512,12,64
transpose t12, t11, 1, 2
reshape t13, t12, 12,512,64
transpose t19, t13, 1, 2
matmul t20, t8, t19
mul t21, t20, 0.125
add t22, t21, mask
softmax t23, t22
matmul t14, t3, h0_attn_wv_w
add t15, t14, h0_attn_wv_b
reshape t16, t15, 1,512,12,64
transpose t17, t16, 1, 2
reshape t18, t17, 12,512,64
matmul t24, t23, t18
reshape t25, t24, 1,12,512,64
transpose t26, t25, 1, 2
reshape t27, t26, 512,768
matmul t28, t27, h0_attn_wo_w
add t29, t28, h0_attn_wo_b
add t30, t2, t29
layernorm t31, t30, h0_ln2_g, h0_ln2_b
matmul t32, t31, h0_mlp_fc_w
add t33, t32, h0_mlp_fc_b
gelu t34, t33
matmul t35, t34, h0_mlp_proj_w
add t36, t35, h0_mlp_proj_b
add t37, t30, t36
layernorm t38, t37, h1_ln1_g, h1_ln1_b
matmul t39, t38, h1_attn_wq_w
add t40, t39, h1_attn_wq_b
reshape t41, t40, 1,512,12,64
transpose t42, t41, 1, 2
reshape t43, t42, 12,512,64
matmul t44, t38, h1_attn_wk_w
add t45, t44, h1_attn_wk_b
reshape t46, t45, 1,512,12,64
transpose t47, t46, 1, 2
reshape t48, t47, 12,512,64
transpose t54, t48, 1, 2
matmul t55, t43, t54
mul t56, t55, 0.125
add t57, t56, mask
softmax t58, t57
matmul t49, t38, h1_attn_wv_w
add t50, t49, h1_attn_wv_b
reshape t51, t50, 1,512,12,64
transpose t52, t51, 1, 2
reshape t53, t52, 12,512,64
matmul t59, t58, t53
reshape t60, t59, 1,12,512,64
transpose t61, t60, 1, 2
reshape t62, t61, 512,768
matmul t63, t62, h1_attn_wo_w
add t64, t63, h1_attn_wo_b
add t65, t37, t64
layernorm t66, t65, h1_ln2_g, h1_ln2_b
matmul t67, t66, h1_mlp_fc_w
add t68, t67, h1_mlp_fc_b
gelu t69, t68
matmul t70, t69, h1_mlp_proj_w
add t71, t70, h1_mlp_proj_b
add t72, t65, t71
layernorm t73, t72, h2_ln1_g, h2_ln1_b
matmul t74, t73, h2_attn_wq_w
add t75, t74, h2_attn_wq_b
reshape t76, t75, 1,512,12,64
transpose t77, t76, 1, 2
reshape t78, t77, 12,512,64
matmul t79, t73, h2_attn_wk_w
add t80, t79, h2_attn_wk_b
reshape t81, t80, 1,512,12,64
transpose t82, t81, 1, 2
reshape t83, t82, 12,512,64
transpose t89, t83, 1, 2
matmul t90, t78, t89
mul t91, t90, 0.125
add t92, t91, mask
softmax t93, t92
matmul t84, t73, h2_attn_wv_w
add t85, t84, h2_attn_wv_b
reshape t86, t85, 1,512,12,64
transpose t87, t86, 1, 2
reshape t88, t87, 12,512,64
matmul t94, t93, t88
reshape t95, t94, 1,12,512,64
transpose t96, t95, 1, 2
reshape t97, t96, 512,768
matmul t98, t97, h2_attn_wo_w
add t99, t98, h2_attn_wo_b
add t100, t72, t99
layernorm t101, t100, h2_ln2_g, h2_ln2_b
matmul t102, t101, h2_mlp_fc_w
add t103, t102, h2_mlp_fc_b
gelu t104, t103
matmul t105, t104, h2_mlp_proj_w
add t106, t105, h2_mlp_proj_b
add t107, t100, t106
layernorm t108, t107, h3_ln1_g, h3_ln1_b
matmul t109, t108, h3_attn_wq_w
add t110, t109, h3_attn_wq_b
reshape t111, t110, 1,512,12,64
transpose t112, t111, 1, 2
reshape t113, t112, 12,512,64
matmul t114, t108, h3_attn_wk_w
add t115, t114, h3_attn_wk_b
reshape t116, t115, 1,512,12,64
transpose t117, t116, 1, 2
reshape t118, t117, 12,512,64
transpose t124, t118, 1, 2
matmul t125, t113, t124
mul t126, t125, 0.125
add t127, t126, mask
softmax t128, t127
matmul t119, t108, h3_attn_wv_w
add t120, t119, h3_attn_wv_b
reshape t121, t120, 1,512,12,64
transpose t122, t121, 1, 2
reshape t123, t122, 12,512,64
matmul t129, t128, t123
reshape t130, t129, 1,12,512,64
transpose t131, t130, 1, 2
reshape t132, t131, 512,768
matmul t133, t132, h3_attn_wo_w
add t134, t133, h3_attn_wo_b
add t135, t107, t134
layernorm t136, t135, h3_ln2_g, h3_ln2_b
matmul t137, t136, h3_mlp_fc_w
add t138, t137, h3_mlp_fc_b
gelu t139, t138
matmul t140, t139, h3_mlp_proj_w
add t141, t140, h3_mlp_proj_b
add t142, t135, t141
layernorm t143, t142, h4_ln1_g, h4_ln1_b
matmul t144, t143, h4_attn_wq_w
add t145, t144, h4_attn_wq_b
reshape t146, t145, 1,512,12,64
transpose t147, t146, 1, 2
reshape t148, t147, 12,512,64
matmul t149, t143, h4_attn_wk_w
add t150, t149, h4_attn_wk_b
reshape t151, t150, 1,512,12,64
transpose t152, t151, 1, 2
reshape t153, t152, 12,512,64
transpose t159, t153, 1, 2
matmul t160, t148, t159
mul t161, t160, 0.125
add t162, t161, mask
softmax t163, t162
matmul t154, t143, h4_attn_wv_w
add t155, t154, h4_attn_wv_b
reshape t156, t155, 1,512,12,64
transpose t157, t156, 1, 2
reshape t158, t157, 12,512,64
matmul t164, t163, t158
reshape t165, t164, 1,12,512,64
transpose t166, t165, 1, 2
reshape t167, t166, 512,768
matmul t168, t167, h4_attn_wo_w
add t169, t168, h4_attn_wo_b
add t170, t142, t169
layernorm t171, t170, h4_ln2_g, h4_ln2_b
matmul t172, t171, h4_mlp_fc_w
add t173, t172, h4_mlp_fc_b
gelu t174, t173
matmul t175, t174, h4_mlp_proj_w
add t176, t175, h4_mlp_proj_b
add t177, t170, t176
layernorm t178, t177, h5_ln1_g, h5_ln1_b
matmul t179, t178, h5_attn_wq_w
add t180, t179, h5_attn_wq_b
reshape t181, t180, 1,512,12,64
transpose t182, t181, 1, 2
reshape t183, t182, 12,512,64
matmul t184, t178, h5_attn_wk_w
add t185, t184, h5_attn_wk_b
reshape t186, t185, 1,512,12,64
transpose t187, t186, 1, 2
reshape t188, t187, 12,512,64
transpose t194, t188, 1, 2
matmul t195, t183, t194
mul t196, t195, 0.125
add t197, t196, mask
softmax t198, t197
matmul t189, t178, h5_attn_wv_w
add t190, t189, h5_attn_wv_b
reshape t191, t190, 1,512,12,64
transpose t192, t191, 1, 2
reshape t193, t192, 12,512,64
matmul t199, t198, t193
reshape t200, t199, 1,12,512,64
transpose t201, t200, 1, 2
reshape t202, t201, 512,768
matmul t203, t202, h5_attn_wo_w
add t204, t203, h5_attn_wo_b
add t205, t177, t204
layernorm t206, t205, h5_ln2_g, h5_ln2_b
matmul t207, t206, h5_mlp_fc_w
add t208, t207, h5_mlp_fc_b
gelu t209, t208
matmul t210, t209, h5_mlp_proj_w
add t211, t210, h5_mlp_proj_b
add t212, t205, t211
layernorm t213, t212, h6_ln1_g, h6_ln1_b
matmul t214, t213, h6_attn_wq_w
add t215, t214, h6_attn_wq_b
reshape t216, t215, 1,512,12,64
transpose t217, t216, 1, 2
reshape t218, t217, 12,512,64
matmul t219, t213, h6_attn_wk_w
add t220, t219, h6_attn_wk_b
reshape t221, t220, 1,512,12,64
transpose t222, t221, 1, 2
reshape t223, t222, 12,512,64
transpose t229, t223, 1, 2
matmul t230, t218, t229
mul t231, t230, 0.125
add t232, t231, mask
softmax t233, t232
matmul t224, t213, h6_attn_wv_w
add t225, t224, h6_attn_wv_b
reshape t226, t225, 1,512,12,64
transpose t227, t226, 1, 2
reshape t228, t227, 12,512,64
matmul t234, t233, t228
reshape t235, t234, 1,12,512,64
transpose t236, t235, 1, 2
reshape t237, t236, 512,768
matmul t238, t237, h6_attn_wo_w
add t239, t238, h6_attn_wo_b
add t240, t212, t239
layernorm t241, t240, h6_ln2_g, h6_ln2_b
matmul t242, t241, h6_mlp_fc_w
add t243, t242, h6_mlp_fc_b
gelu t244, t243
matmul t245, t244, h6_mlp_proj_w
add t246, t245, h6_mlp_proj_b
add t247, t240, t246
layernorm t248, t247, h7_ln1_g, h7_ln1_b
matmul t249, t248, h7_attn_wq_w
add t250, t249, h7_attn_wq_b
reshape t251, t250, 1,512,12,64
transpose t252, t251, 1, 2
reshape t253, t252, 12,512,64
matmul t254, t248, h7_attn_wk_w
add t255, t254, h7_attn_wk_b
reshape t256, t255, 1,512,12,64
transpose t257, t256, 1, 2
reshape t258, t257, 12,512,64
transpose t264, t258, 1, 2
matmul t265, t253, t264
mul t266, t265, 0.125
add t267, t266, mask
softmax t268, t267
matmul t259, t248, h7_attn_wv_w
add t260, t259, h7_attn_wv_b
reshape t261, t260, 1,512,12,64
transpose t262, t261, 1, 2
reshape t263, t262, 12,512,64
matmul t269, t268, t263
reshape t270, t269, 1,12,512,64
transpose t271, t270, 1, 2
reshape t272, t271, 512,768
matmul t273, t272, h7_attn_wo_w
add t274, t273, h7_attn_wo_b
add t275, t247, t274
layernorm t276, t275, h7_ln2_g, h7_ln2_b
matmul t277, t276, h7_mlp_fc_w
add t278, t277, h7_mlp_fc_b
gelu t279, t278
matmul t280, t279, h7_mlp_proj_w
add t281, t280, h7_mlp_proj_b
add t282, t275, t281
layernorm t283, t282, h8_ln1_g, h8_ln1_b
matmul t284, t283, h8_attn_wq_w
add t285, t284, h8_attn_wq_b
reshape t286, t285, 1,512,12,64
transpose t287, t286, 1, 2
reshape t288, t287, 12,512,64
matmul t289, t283, h8_attn_wk_w
add t290, t289, h8_attn_wk_b
reshape t291, t290, 1,512,12,64
transpose t292, t291, 1, 2
reshape t293, t292, 12,512,64
transpose t299, t293, 1, 2
matmul t300, t288, t299
mul t301, t300, 0.125
add t302, t301, mask
softmax t303, t302
matmul t294, t283, h8_attn_wv_w
add t295, t294, h8_attn_wv_b
reshape t296, t295, 1,512,12,64
transpose t297, t296, 1, 2
reshape t298, t297, 12,512,64
matmul t304, t303, t298
reshape t305, t304, 1,12,512,64
transpose t306, t305, 1, 2
reshape t307, t306, 512,768
matmul t308, t307, h8_attn_wo_w
add t309, t308, h8_attn_wo_b
add t310, t282, t309
layernorm t311, t310, h8_ln2_g, h8_ln2_b
matmul t312, t311, h8_mlp_fc_w
add t313, t312, h8_mlp_fc_b
gelu t314, t313
matmul t315, t314, h8_mlp_proj_w
add t316, t315, h8_mlp_proj_b
add t317, t310, t316
layernorm t318, t317, h9_ln1_g, h9_ln1_b
matmul t319, t318, h9_attn_wq_w
add t320, t319, h9_attn_wq_b
reshape t321, t320, 1,512,12,64
transpose t322, t321, 1, 2
reshape t323, t322, 12,512,64
matmul t324, t318, h9_attn_wk_w
add t325, t324, h9_attn_wk_b
reshape t326, t325, 1,512,12,64
transpose t327, t326, 1, 2
reshape t328, t327, 12,512,64
transpose t334, t328, 1, 2
matmul t335, t323, t334
mul t336, t335, 0.125
add t337, t336, mask
softmax t338, t337
matmul t329, t318, h9_attn_wv_w
add t330, t329, h9_attn_wv_b
reshape t331, t330, 1,512,12,64
transpose t332, t331, 1, 2
reshape t333, t332, 12,512,64
matmul t339, t338, t333
reshape t340, t339, 1,12,512,64
transpose t341, t340, 1, 2
reshape t342, t341, 512,768
matmul t343, t342, h9_attn_wo_w
add t344, t343, h9_attn_wo_b
add t345, t317, t344
layernorm t346, t345, h9_ln2_g, h9_ln2_b
matmul t347, t346, h9_mlp_fc_w
add t348, t347, h9_mlp_fc_b
gelu t349, t348
matmul t350, t349, h9_mlp_proj_w
add t351, t350, h9_mlp_proj_b
add t352, t345, t351
layernorm t353, t352, h10_ln1_g, h10_ln1_b
matmul t354, t353, h10_attn_wq_w
add t355, t354, h10_attn_wq_b
reshape t356, t355, 1,512,12,64
transpose t357, t356, 1, 2
reshape t358, t357, 12,512,64
matmul t359, t353, h10_attn_wk_w
add t360, t359, h10_attn_wk_b
reshape t361, t360, 1,512,12,64
transpose t362, t361, 1, 2
reshape t363, t362, 12,512,64
transpose t369, t363, 1, 2
matmul t370, t358, t369
mul t371, t370, 0.125
add t372, t371, mask
softmax t373, t372
matmul t364, t353, h10_attn_wv_w
add t365, t364, h10_attn_wv_b
reshape t366, t365, 1,512,12,64
transpose t367, t366, 1, 2
reshape t368, t367, 12,512,64
matmul t374, t373, t368
reshape t375, t374, 1,12,512,64
transpose t376, t375, 1, 2
reshape t377, t376, 512,768
matmul t378, t377, h10_attn_wo_w
add t379, t378, h10_attn_wo_b
add t380, t352, t379
layernorm t381, t380, h10_ln2_g, h10_ln2_b
matmul t382, t381, h10_mlp_fc_w
add t383, t382, h10_mlp_fc_b
gelu t384, t383
matmul t385, t384, h10_mlp_proj_w
add t386, t385, h10_mlp_proj_b
add t387, t380, t386
layernorm t388, t387, h11_ln1_g, h11_ln1_b
matmul t389, t388, h11_attn_wq_w
add t390, t389, h11_attn_wq_b
reshape t391, t390, 1,512,12,64
transpose t392, t391, 1, 2
reshape t393, t392, 12,512,64
matmul t394, t388, h11_attn_wk_w
add t395, t394, h11_attn_wk_b
reshape t396, t395, 1,512,12,64
transpose t397, t396, 1, 2
reshape t398, t397, 12,512,64
transpose t404, t398, 1, 2
matmul t405, t393, t404
mul t406, t405, 0.125
add t407, t406, mask
softmax t408, t407
matmul t399, t388, h11_attn_wv_w
add t400, t399, h11_attn_wv_b
reshape t401, t400, 1,512,12,64
transpose t402, t401, 1, 2
reshape t403, t402, 12,512,64
matmul t409, t408, t403
reshape t410, t409, 1,12,512,64
transpose t411, t410, 1, 2
reshape t412, t411, 512,768
matmul t413, t412, h11_attn_wo_w
add t414, t413, h11_attn_wo_b
add t415, t387, t414
layernorm t416, t415, h11_ln2_g, h11_ln2_b
matmul t417, t416, h11_mlp_fc_w
add t418, t417, h11_mlp_fc_b
gelu t419, t418
matmul t420, t419, h11_mlp_proj_w
add t421, t420, h11_mlp_proj_b
add t422, t415, t421
layernorm t423, t422, ln_f_g, ln_f_b
transpose t424, wte, 0, 1
matmul t425, t423, t424
cross_entropy t426, g_t425, t425, targets

# === BACKWARD PASS ===
transpose t427, t424
matmul t428, g_t425, t427
assign g_t423, t428
transpose t429, t423
matmul t430, t429, g_t425
assign g_t424, t430
transpose t431, g_t424, 0, 1
assign g_wte, t431
layernorm_grad t432, t433, t434, g_t423, t422, ln_f_g
assign g_t422, t432
assign g_ln_f_g, t433
assign g_ln_f_b, t434
assign g_t415, g_t422
assign g_t421, g_t422
assign g_t420, g_t421
sum t435, g_t421, 0
reshape t436, t435, 768
assign g_h11_mlp_proj_b, t436
transpose t437, h11_mlp_proj_w
matmul t438, g_t420, t437
assign g_t419, t438
transpose t439, t419
matmul t440, t439, g_t420
assign g_h11_mlp_proj_w, t440
gelu_grad t441, g_t419, t418
assign g_t418, t441
assign g_t417, g_t418
sum t442, g_t418, 0
reshape t443, t442, 3072
assign g_h11_mlp_fc_b, t443
transpose t444, h11_mlp_fc_w
matmul t445, g_t417, t444
assign g_t416, t445
transpose t446, t416
matmul t447, t446, g_t417
assign g_h11_mlp_fc_w, t447
layernorm_grad t448, t449, t450, g_t416, t415, h11_ln2_g
add g_t415, g_t415, t448
assign g_h11_ln2_g, t449
assign g_h11_ln2_b, t450
assign g_t387, g_t415
assign g_t414, g_t415
assign g_t413, g_t414
sum t451, g_t414, 0
reshape t452, t451, 768
assign g_h11_attn_wo_b, t452
transpose t453, h11_attn_wo_w
matmul t454, g_t413, t453
assign g_t412, t454
transpose t455, t412
matmul t456, t455, g_t413
assign g_h11_attn_wo_w, t456
reshape t457, g_t412, 1,512,12,64
assign g_t411, t457
transpose t458, g_t411, 1, 2
assign g_t410, t458
reshape t459, g_t410, 12,512,64
assign g_t409, t459
transpose t460, t403
matmul t461, g_t409, t460
assign g_t408, t461
transpose t462, t408
matmul t463, t462, g_t409
assign g_t403, t463
reshape t464, g_t403, 1,12,512,64
assign g_t402, t464
transpose t465, g_t402, 1, 2
assign g_t401, t465
reshape t466, g_t401, 512,768
assign g_t400, t466
assign g_t399, g_t400
sum t467, g_t400, 0
reshape t468, t467, 768
assign g_h11_attn_wv_b, t468
transpose t469, h11_attn_wv_w
matmul t470, g_t399, t469
assign g_t388, t470
transpose t471, t388
matmul t472, t471, g_t399
assign g_h11_attn_wv_w, t472
softmax_grad t473, g_t408, t408
assign g_t407, t473
assign g_t406, g_t407
mul t474, g_t406, 0.125
assign g_t405, t474
transpose t475, t404
matmul t476, g_t405, t475
assign g_t393, t476
transpose t477, t393
matmul t478, t477, g_t405
assign g_t404, t478
transpose t479, g_t404, 1, 2
assign g_t398, t479
reshape t480, g_t398, 1,12,512,64
assign g_t397, t480
transpose t481, g_t397, 1, 2
assign g_t396, t481
reshape t482, g_t396, 512,768
assign g_t395, t482
assign g_t394, g_t395
sum t483, g_t395, 0
reshape t484, t483, 768
assign g_h11_attn_wk_b, t484
transpose t485, h11_attn_wk_w
matmul t486, g_t394, t485
add g_t388, g_t388, t486
transpose t487, t388
matmul t488, t487, g_t394
assign g_h11_attn_wk_w, t488
reshape t489, g_t393, 1,12,512,64
assign g_t392, t489
transpose t490, g_t392, 1, 2
assign g_t391, t490
reshape t491, g_t391, 512,768
assign g_t390, t491
assign g_t389, g_t390
sum t492, g_t390, 0
reshape t493, t492, 768
assign g_h11_attn_wq_b, t493
transpose t494, h11_attn_wq_w
matmul t495, g_t389, t494
add g_t388, g_t388, t495
transpose t496, t388
matmul t497, t496, g_t389
assign g_h11_attn_wq_w, t497
layernorm_grad t498, t499, t500, g_t388, t387, h11_ln1_g
add g_t387, g_t387, t498
assign g_h11_ln1_g, t499
assign g_h11_ln1_b, t500
assign g_t380, g_t387
assign g_t386, g_t387
assign g_t385, g_t386
sum t501, g_t386, 0
reshape t502, t501, 768
assign g_h10_mlp_proj_b, t502
transpose t503, h10_mlp_proj_w
matmul t504, g_t385, t503
assign g_t384, t504
transpose t505, t384
matmul t506, t505, g_t385
assign g_h10_mlp_proj_w, t506
gelu_grad t507, g_t384, t383
assign g_t383, t507
assign g_t382, g_t383
sum t508, g_t383, 0
reshape t509, t508, 3072
assign g_h10_mlp_fc_b, t509
transpose t510, h10_mlp_fc_w
matmul t511, g_t382, t510
assign g_t381, t511
transpose t512, t381
matmul t513, t512, g_t382
assign g_h10_mlp_fc_w, t513
layernorm_grad t514, t515, t516, g_t381, t380, h10_ln2_g
add g_t380, g_t380, t514
assign g_h10_ln2_g, t515
assign g_h10_ln2_b, t516
assign g_t352, g_t380
assign g_t379, g_t380
assign g_t378, g_t379
sum t517, g_t379, 0
reshape t518, t517, 768
assign g_h10_attn_wo_b, t518
transpose t519, h10_attn_wo_w
matmul t520, g_t378, t519
assign g_t377, t520
transpose t521, t377
matmul t522, t521, g_t378
assign g_h10_attn_wo_w, t522
reshape t523, g_t377, 1,512,12,64
assign g_t376, t523
transpose t524, g_t376, 1, 2
assign g_t375, t524
reshape t525, g_t375, 12,512,64
assign g_t374, t525
transpose t526, t368
matmul t527, g_t374, t526
assign g_t373, t527
transpose t528, t373
matmul t529, t528, g_t374
assign g_t368, t529
reshape t530, g_t368, 1,12,512,64
assign g_t367, t530
transpose t531, g_t367, 1, 2
assign g_t366, t531
reshape t532, g_t366, 512,768
assign g_t365, t532
assign g_t364, g_t365
sum t533, g_t365, 0
reshape t534, t533, 768
assign g_h10_attn_wv_b, t534
transpose t535, h10_attn_wv_w
matmul t536, g_t364, t535
assign g_t353, t536
transpose t537, t353
matmul t538, t537, g_t364
assign g_h10_attn_wv_w, t538
softmax_grad t539, g_t373, t373
assign g_t372, t539
assign g_t371, g_t372
mul t540, g_t371, 0.125
assign g_t370, t540
transpose t541, t369
matmul t542, g_t370, t541
assign g_t358, t542
transpose t543, t358
matmul t544, t543, g_t370
assign g_t369, t544
transpose t545, g_t369, 1, 2
assign g_t363, t545
reshape t546, g_t363, 1,12,512,64
assign g_t362, t546
transpose t547, g_t362, 1, 2
assign g_t361, t547
reshape t548, g_t361, 512,768
assign g_t360, t548
assign g_t359, g_t360
sum t549, g_t360, 0
reshape t550, t549, 768
assign g_h10_attn_wk_b, t550
transpose t551, h10_attn_wk_w
matmul t552, g_t359, t551
add g_t353, g_t353, t552
transpose t553, t353
matmul t554, t553, g_t359
assign g_h10_attn_wk_w, t554
reshape t555, g_t358, 1,12,512,64
assign g_t357, t555
transpose t556, g_t357, 1, 2
assign g_t356, t556
reshape t557, g_t356, 512,768
assign g_t355, t557
assign g_t354, g_t355
sum t558, g_t355, 0
reshape t559, t558, 768
assign g_h10_attn_wq_b, t559
transpose t560, h10_attn_wq_w
matmul t561, g_t354, t560
add g_t353, g_t353, t561
transpose t562, t353
matmul t563, t562, g_t354
assign g_h10_attn_wq_w, t563
layernorm_grad t564, t565, t566, g_t353, t352, h10_ln1_g
add g_t352, g_t352, t564
assign g_h10_ln1_g, t565
assign g_h10_ln1_b, t566
assign g_t345, g_t352
assign g_t351, g_t352
assign g_t350, g_t351
sum t567, g_t351, 0
reshape t568, t567, 768
assign g_h9_mlp_proj_b, t568
transpose t569, h9_mlp_proj_w
matmul t570, g_t350, t569
assign g_t349, t570
transpose t571, t349
matmul t572, t571, g_t350
assign g_h9_mlp_proj_w, t572
gelu_grad t573, g_t349, t348
assign g_t348, t573
assign g_t347, g_t348
sum t574, g_t348, 0
reshape t575, t574, 3072
assign g_h9_mlp_fc_b, t575
transpose t576, h9_mlp_fc_w
matmul t577, g_t347, t576
assign g_t346, t577
transpose t578, t346
matmul t579, t578, g_t347
assign g_h9_mlp_fc_w, t579
layernorm_grad t580, t581, t582, g_t346, t345, h9_ln2_g
add g_t345, g_t345, t580
assign g_h9_ln2_g, t581
assign g_h9_ln2_b, t582
assign g_t317, g_t345
assign g_t344, g_t345
assign g_t343, g_t344
sum t583, g_t344, 0
reshape t584, t583, 768
assign g_h9_attn_wo_b, t584
transpose t585, h9_attn_wo_w
matmul t586, g_t343, t585
assign g_t342, t586
transpose t587, t342
matmul t588, t587, g_t343
assign g_h9_attn_wo_w, t588
reshape t589, g_t342, 1,512,12,64
assign g_t341, t589
transpose t590, g_t341, 1, 2
assign g_t340, t590
reshape t591, g_t340, 12,512,64
assign g_t339, t591
transpose t592, t333
matmul t593, g_t339, t592
assign g_t338, t593
transpose t594, t338
matmul t595, t594, g_t339
assign g_t333, t595
reshape t596, g_t333, 1,12,512,64
assign g_t332, t596
transpose t597, g_t332, 1, 2
assign g_t331, t597
reshape t598, g_t331, 512,768
assign g_t330, t598
assign g_t329, g_t330
sum t599, g_t330, 0
reshape t600, t599, 768
assign g_h9_attn_wv_b, t600
transpose t601, h9_attn_wv_w
matmul t602, g_t329, t601
assign g_t318, t602
transpose t603, t318
matmul t604, t603, g_t329
assign g_h9_attn_wv_w, t604
softmax_grad t605, g_t338, t338
assign g_t337, t605
assign g_t336, g_t337
mul t606, g_t336, 0.125
assign g_t335, t606
transpose t607, t334
matmul t608, g_t335, t607
assign g_t323, t608
transpose t609, t323
matmul t610, t609, g_t335
assign g_t334, t610
transpose t611, g_t334, 1, 2
assign g_t328, t611
reshape t612, g_t328, 1,12,512,64
assign g_t327, t612
transpose t613, g_t327, 1, 2
assign g_t326, t613
reshape t614, g_t326, 512,768
assign g_t325, t614
assign g_t324, g_t325
sum t615, g_t325, 0
reshape t616, t615, 768
assign g_h9_attn_wk_b, t616
transpose t617, h9_attn_wk_w
matmul t618, g_t324, t617
add g_t318, g_t318, t618
transpose t619, t318
matmul t620, t619, g_t324
assign g_h9_attn_wk_w, t620
reshape t621, g_t323, 1,12,512,64
assign g_t322, t621
transpose t622, g_t322, 1, 2
assign g_t321, t622
reshape t623, g_t321, 512,768
assign g_t320, t623
assign g_t319, g_t320
sum t624, g_t320, 0
reshape t625, t624, 768
assign g_h9_attn_wq_b, t625
transpose t626, h9_attn_wq_w
matmul t627, g_t319, t626
add g_t318, g_t318, t627
transpose t628, t318
matmul t629, t628, g_t319
assign g_h9_attn_wq_w, t629
layernorm_grad t630, t631, t632, g_t318, t317, h9_ln1_g
add g_t317, g_t317, t630
assign g_h9_ln1_g, t631
assign g_h9_ln1_b, t632
assign g_t310, g_t317
assign g_t316, g_t317
assign g_t315, g_t316
sum t633, g_t316, 0
reshape t634, t633, 768
assign g_h8_mlp_proj_b, t634
transpose t635, h8_mlp_proj_w
matmul t636, g_t315, t635
assign g_t314, t636
transpose t637, t314
matmul t638, t637, g_t315
assign g_h8_mlp_proj_w, t638
gelu_grad t639, g_t314, t313
assign g_t313, t639
assign g_t312, g_t313
sum t640, g_t313, 0
reshape t641, t640, 3072
assign g_h8_mlp_fc_b, t641
transpose t642, h8_mlp_fc_w
matmul t643, g_t312, t642
assign g_t311, t643
transpose t644, t311
matmul t645, t644, g_t312
assign g_h8_mlp_fc_w, t645
layernorm_grad t646, t647, t648, g_t311, t310, h8_ln2_g
add g_t310, g_t310, t646
assign g_h8_ln2_g, t647
assign g_h8_ln2_b, t648
assign g_t282, g_t310
assign g_t309, g_t310
assign g_t308, g_t309
sum t649, g_t309, 0
reshape t650, t649, 768
assign g_h8_attn_wo_b, t650
transpose t651, h8_attn_wo_w
matmul t652, g_t308, t651
assign g_t307, t652
transpose t653, t307
matmul t654, t653, g_t308
assign g_h8_attn_wo_w, t654
reshape t655, g_t307, 1,512,12,64
assign g_t306, t655
transpose t656, g_t306, 1, 2
assign g_t305, t656
reshape t657, g_t305, 12,512,64
assign g_t304, t657
transpose t658, t298
matmul t659, g_t304, t658
assign g_t303, t659
transpose t660, t303
matmul t661, t660, g_t304
assign g_t298, t661
reshape t662, g_t298, 1,12,512,64
assign g_t297, t662
transpose t663, g_t297, 1, 2
assign g_t296, t663
reshape t664, g_t296, 512,768
assign g_t295, t664
assign g_t294, g_t295
sum t665, g_t295, 0
reshape t666, t665, 768
assign g_h8_attn_wv_b, t666
transpose t667, h8_attn_wv_w
matmul t668, g_t294, t667
assign g_t283, t668
transpose t669, t283
matmul t670, t669, g_t294
assign g_h8_attn_wv_w, t670
softmax_grad t671, g_t303, t303
assign g_t302, t671
assign g_t301, g_t302
mul t672, g_t301, 0.125
assign g_t300, t672
transpose t673, t299
matmul t674, g_t300, t673
assign g_t288, t674
transpose t675, t288
matmul t676, t675, g_t300
assign g_t299, t676
transpose t677, g_t299, 1, 2
assign g_t293, t677
reshape t678, g_t293, 1,12,512,64
assign g_t292, t678
transpose t679, g_t292, 1, 2
assign g_t291, t679
reshape t680, g_t291, 512,768
assign g_t290, t680
assign g_t289, g_t290
sum t681, g_t290, 0
reshape t682, t681, 768
assign g_h8_attn_wk_b, t682
transpose t683, h8_attn_wk_w
matmul t684, g_t289, t683
add g_t283, g_t283, t684
transpose t685, t283
matmul t686, t685, g_t289
assign g_h8_attn_wk_w, t686
reshape t687, g_t288, 1,12,512,64
assign g_t287, t687
transpose t688, g_t287, 1, 2
assign g_t286, t688
reshape t689, g_t286, 512,768
assign g_t285, t689
assign g_t284, g_t285
sum t690, g_t285, 0
reshape t691, t690, 768
assign g_h8_attn_wq_b, t691
transpose t692, h8_attn_wq_w
matmul t693, g_t284, t692
add g_t283, g_t283, t693
transpose t694, t283
matmul t695, t694, g_t284
assign g_h8_attn_wq_w, t695
layernorm_grad t696, t697, t698, g_t283, t282, h8_ln1_g
add g_t282, g_t282, t696
assign g_h8_ln1_g, t697
assign g_h8_ln1_b, t698
assign g_t275, g_t282
assign g_t281, g_t282
assign g_t280, g_t281
sum t699, g_t281, 0
reshape t700, t699, 768
assign g_h7_mlp_proj_b, t700
transpose t701, h7_mlp_proj_w
matmul t702, g_t280, t701
assign g_t279, t702
transpose t703, t279
matmul t704, t703, g_t280
assign g_h7_mlp_proj_w, t704
gelu_grad t705, g_t279, t278
assign g_t278, t705
assign g_t277, g_t278
sum t706, g_t278, 0
reshape t707, t706, 3072
assign g_h7_mlp_fc_b, t707
transpose t708, h7_mlp_fc_w
matmul t709, g_t277, t708
assign g_t276, t709
transpose t710, t276
matmul t711, t710, g_t277
assign g_h7_mlp_fc_w, t711
layernorm_grad t712, t713, t714, g_t276, t275, h7_ln2_g
add g_t275, g_t275, t712
assign g_h7_ln2_g, t713
assign g_h7_ln2_b, t714
assign g_t247, g_t275
assign g_t274, g_t275
assign g_t273, g_t274
sum t715, g_t274, 0
reshape t716, t715, 768
assign g_h7_attn_wo_b, t716
transpose t717, h7_attn_wo_w
matmul t718, g_t273, t717
assign g_t272, t718
transpose t719, t272
matmul t720, t719, g_t273
assign g_h7_attn_wo_w, t720
reshape t721, g_t272, 1,512,12,64
assign g_t271, t721
transpose t722, g_t271, 1, 2
assign g_t270, t722
reshape t723, g_t270, 12,512,64
assign g_t269, t723
transpose t724, t263
matmul t725, g_t269, t724
assign g_t268, t725
transpose t726, t268
matmul t727, t726, g_t269
assign g_t263, t727
reshape t728, g_t263, 1,12,512,64
assign g_t262, t728
transpose t729, g_t262, 1, 2
assign g_t261, t729
reshape t730, g_t261, 512,768
assign g_t260, t730
assign g_t259, g_t260
sum t731, g_t260, 0
reshape t732, t731, 768
assign g_h7_attn_wv_b, t732
transpose t733, h7_attn_wv_w
matmul t734, g_t259, t733
assign g_t248, t734
transpose t735, t248
matmul t736, t735, g_t259
assign g_h7_attn_wv_w, t736
softmax_grad t737, g_t268, t268
assign g_t267, t737
assign g_t266, g_t267
mul t738, g_t266, 0.125
assign g_t265, t738
transpose t739, t264
matmul t740, g_t265, t739
assign g_t253, t740
transpose t741, t253
matmul t742, t741, g_t265
assign g_t264, t742
transpose t743, g_t264, 1, 2
assign g_t258, t743
reshape t744, g_t258, 1,12,512,64
assign g_t257, t744
transpose t745, g_t257, 1, 2
assign g_t256, t745
reshape t746, g_t256, 512,768
assign g_t255, t746
assign g_t254, g_t255
sum t747, g_t255, 0
reshape t748, t747, 768
assign g_h7_attn_wk_b, t748
transpose t749, h7_attn_wk_w
matmul t750, g_t254, t749
add g_t248, g_t248, t750
transpose t751, t248
matmul t752, t751, g_t254
assign g_h7_attn_wk_w, t752
reshape t753, g_t253, 1,12,512,64
assign g_t252, t753
transpose t754, g_t252, 1, 2
assign g_t251, t754
reshape t755, g_t251, 512,768
assign g_t250, t755
assign g_t249, g_t250
sum t756, g_t250, 0
reshape t757, t756, 768
assign g_h7_attn_wq_b, t757
transpose t758, h7_attn_wq_w
matmul t759, g_t249, t758
add g_t248, g_t248, t759
transpose t760, t248
matmul t761, t760, g_t249
assign g_h7_attn_wq_w, t761
layernorm_grad t762, t763, t764, g_t248, t247, h7_ln1_g
add g_t247, g_t247, t762
assign g_h7_ln1_g, t763
assign g_h7_ln1_b, t764
assign g_t240, g_t247
assign g_t246, g_t247
assign g_t245, g_t246
sum t765, g_t246, 0
reshape t766, t765, 768
assign g_h6_mlp_proj_b, t766
transpose t767, h6_mlp_proj_w
matmul t768, g_t245, t767
assign g_t244, t768
transpose t769, t244
matmul t770, t769, g_t245
assign g_h6_mlp_proj_w, t770
gelu_grad t771, g_t244, t243
assign g_t243, t771
assign g_t242, g_t243
sum t772, g_t243, 0
reshape t773, t772, 3072
assign g_h6_mlp_fc_b, t773
transpose t774, h6_mlp_fc_w
matmul t775, g_t242, t774
assign g_t241, t775
transpose t776, t241
matmul t777, t776, g_t242
assign g_h6_mlp_fc_w, t777
layernorm_grad t778, t779, t780, g_t241, t240, h6_ln2_g
add g_t240, g_t240, t778
assign g_h6_ln2_g, t779
assign g_h6_ln2_b, t780
assign g_t212, g_t240
assign g_t239, g_t240
assign g_t238, g_t239
sum t781, g_t239, 0
reshape t782, t781, 768
assign g_h6_attn_wo_b, t782
transpose t783, h6_attn_wo_w
matmul t784, g_t238, t783
assign g_t237, t784
transpose t785, t237
matmul t786, t785, g_t238
assign g_h6_attn_wo_w, t786
reshape t787, g_t237, 1,512,12,64
assign g_t236, t787
transpose t788, g_t236, 1, 2
assign g_t235, t788
reshape t789, g_t235, 12,512,64
assign g_t234, t789
transpose t790, t228
matmul t791, g_t234, t790
assign g_t233, t791
transpose t792, t233
matmul t793, t792, g_t234
assign g_t228, t793
reshape t794, g_t228, 1,12,512,64
assign g_t227, t794
transpose t795, g_t227, 1, 2
assign g_t226, t795
reshape t796, g_t226, 512,768
assign g_t225, t796
assign g_t224, g_t225
sum t797, g_t225, 0
reshape t798, t797, 768
assign g_h6_attn_wv_b, t798
transpose t799, h6_attn_wv_w
matmul t800, g_t224, t799
assign g_t213, t800
transpose t801, t213
matmul t802, t801, g_t224
assign g_h6_attn_wv_w, t802
softmax_grad t803, g_t233, t233
assign g_t232, t803
assign g_t231, g_t232
mul t804, g_t231, 0.125
assign g_t230, t804
transpose t805, t229
matmul t806, g_t230, t805
assign g_t218, t806
transpose t807, t218
matmul t808, t807, g_t230
assign g_t229, t808
transpose t809, g_t229, 1, 2
assign g_t223, t809
reshape t810, g_t223, 1,12,512,64
assign g_t222, t810
transpose t811, g_t222, 1, 2
assign g_t221, t811
reshape t812, g_t221, 512,768
assign g_t220, t812
assign g_t219, g_t220
sum t813, g_t220, 0
reshape t814, t813, 768
assign g_h6_attn_wk_b, t814
transpose t815, h6_attn_wk_w
matmul t816, g_t219, t815
add g_t213, g_t213, t816
transpose t817, t213
matmul t818, t817, g_t219
assign g_h6_attn_wk_w, t818
reshape t819, g_t218, 1,12,512,64
assign g_t217, t819
transpose t820, g_t217, 1, 2
assign g_t216, t820
reshape t821, g_t216, 512,768
assign g_t215, t821
assign g_t214, g_t215
sum t822, g_t215, 0
reshape t823, t822, 768
assign g_h6_attn_wq_b, t823
transpose t824, h6_attn_wq_w
matmul t825, g_t214, t824
add g_t213, g_t213, t825
transpose t826, t213
matmul t827, t826, g_t214
assign g_h6_attn_wq_w, t827
layernorm_grad t828, t829, t830, g_t213, t212, h6_ln1_g
add g_t212, g_t212, t828
assign g_h6_ln1_g, t829
assign g_h6_ln1_b, t830
assign g_t205, g_t212
assign g_t211, g_t212
assign g_t210, g_t211
sum t831, g_t211, 0
reshape t832, t831, 768
assign g_h5_mlp_proj_b, t832
transpose t833, h5_mlp_proj_w
matmul t834, g_t210, t833
assign g_t209, t834
transpose t835, t209
matmul t836, t835, g_t210
assign g_h5_mlp_proj_w, t836
gelu_grad t837, g_t209, t208
assign g_t208, t837
assign g_t207, g_t208
sum t838, g_t208, 0
reshape t839, t838, 3072
assign g_h5_mlp_fc_b, t839
transpose t840, h5_mlp_fc_w
matmul t841, g_t207, t840
assign g_t206, t841
transpose t842, t206
matmul t843, t842, g_t207
assign g_h5_mlp_fc_w, t843
layernorm_grad t844, t845, t846, g_t206, t205, h5_ln2_g
add g_t205, g_t205, t844
assign g_h5_ln2_g, t845
assign g_h5_ln2_b, t846
assign g_t177, g_t205
assign g_t204, g_t205
assign g_t203, g_t204
sum t847, g_t204, 0
reshape t848, t847, 768
assign g_h5_attn_wo_b, t848
transpose t849, h5_attn_wo_w
matmul t850, g_t203, t849
assign g_t202, t850
transpose t851, t202
matmul t852, t851, g_t203
assign g_h5_attn_wo_w, t852
reshape t853, g_t202, 1,512,12,64
assign g_t201, t853
transpose t854, g_t201, 1, 2
assign g_t200, t854
reshape t855, g_t200, 12,512,64
assign g_t199, t855
transpose t856, t193
matmul t857, g_t199, t856
assign g_t198, t857
transpose t858, t198
matmul t859, t858, g_t199
assign g_t193, t859
reshape t860, g_t193, 1,12,512,64
assign g_t192, t860
transpose t861, g_t192, 1, 2
assign g_t191, t861
reshape t862, g_t191, 512,768
assign g_t190, t862
assign g_t189, g_t190
sum t863, g_t190, 0
reshape t864, t863, 768
assign g_h5_attn_wv_b, t864
transpose t865, h5_attn_wv_w
matmul t866, g_t189, t865
assign g_t178, t866
transpose t867, t178
matmul t868, t867, g_t189
assign g_h5_attn_wv_w, t868
softmax_grad t869, g_t198, t198
assign g_t197, t869
assign g_t196, g_t197
mul t870, g_t196, 0.125
assign g_t195, t870
transpose t871, t194
matmul t872, g_t195, t871
assign g_t183, t872
transpose t873, t183
matmul t874, t873, g_t195
assign g_t194, t874
transpose t875, g_t194, 1, 2
assign g_t188, t875
reshape t876, g_t188, 1,12,512,64
assign g_t187, t876
transpose t877, g_t187, 1, 2
assign g_t186, t877
reshape t878, g_t186, 512,768
assign g_t185, t878
assign g_t184, g_t185
sum t879, g_t185, 0
reshape t880, t879, 768
assign g_h5_attn_wk_b, t880
transpose t881, h5_attn_wk_w
matmul t882, g_t184, t881
add g_t178, g_t178, t882
transpose t883, t178
matmul t884, t883, g_t184
assign g_h5_attn_wk_w, t884
reshape t885, g_t183, 1,12,512,64
assign g_t182, t885
transpose t886, g_t182, 1, 2
assign g_t181, t886
reshape t887, g_t181, 512,768
assign g_t180, t887
assign g_t179, g_t180
sum t888, g_t180, 0
reshape t889, t888, 768
assign g_h5_attn_wq_b, t889
transpose t890, h5_attn_wq_w
matmul t891, g_t179, t890
add g_t178, g_t178, t891
transpose t892, t178
matmul t893, t892, g_t179
assign g_h5_attn_wq_w, t893
layernorm_grad t894, t895, t896, g_t178, t177, h5_ln1_g
add g_t177, g_t177, t894
assign g_h5_ln1_g, t895
assign g_h5_ln1_b, t896
assign g_t170, g_t177
assign g_t176, g_t177
assign g_t175, g_t176
sum t897, g_t176, 0
reshape t898, t897, 768
assign g_h4_mlp_proj_b, t898
transpose t899, h4_mlp_proj_w
matmul t900, g_t175, t899
assign g_t174, t900
transpose t901, t174
matmul t902, t901, g_t175
assign g_h4_mlp_proj_w, t902
gelu_grad t903, g_t174, t173
assign g_t173, t903
assign g_t172, g_t173
sum t904, g_t173, 0
reshape t905, t904, 3072
assign g_h4_mlp_fc_b, t905
transpose t906, h4_mlp_fc_w
matmul t907, g_t172, t906
assign g_t171, t907
transpose t908, t171
matmul t909, t908, g_t172
assign g_h4_mlp_fc_w, t909
layernorm_grad t910, t911, t912, g_t171, t170, h4_ln2_g
add g_t170, g_t170, t910
assign g_h4_ln2_g, t911
assign g_h4_ln2_b, t912
assign g_t142, g_t170
assign g_t169, g_t170
assign g_t168, g_t169
sum t913, g_t169, 0
reshape t914, t913, 768
assign g_h4_attn_wo_b, t914
transpose t915, h4_attn_wo_w
matmul t916, g_t168, t915
assign g_t167, t916
transpose t917, t167
matmul t918, t917, g_t168
assign g_h4_attn_wo_w, t918
reshape t919, g_t167, 1,512,12,64
assign g_t166, t919
transpose t920, g_t166, 1, 2
assign g_t165, t920
reshape t921, g_t165, 12,512,64
assign g_t164, t921
transpose t922, t158
matmul t923, g_t164, t922
assign g_t163, t923
transpose t924, t163
matmul t925, t924, g_t164
assign g_t158, t925
reshape t926, g_t158, 1,12,512,64
assign g_t157, t926
transpose t927, g_t157, 1, 2
assign g_t156, t927
reshape t928, g_t156, 512,768
assign g_t155, t928
assign g_t154, g_t155
sum t929, g_t155, 0
reshape t930, t929, 768
assign g_h4_attn_wv_b, t930
transpose t931, h4_attn_wv_w
matmul t932, g_t154, t931
assign g_t143, t932
transpose t933, t143
matmul t934, t933, g_t154
assign g_h4_attn_wv_w, t934
softmax_grad t935, g_t163, t163
assign g_t162, t935
assign g_t161, g_t162
mul t936, g_t161, 0.125
assign g_t160, t936
transpose t937, t159
matmul t938, g_t160, t937
assign g_t148, t938
transpose t939, t148
matmul t940, t939, g_t160
assign g_t159, t940
transpose t941, g_t159, 1, 2
assign g_t153, t941
reshape t942, g_t153, 1,12,512,64
assign g_t152, t942
transpose t943, g_t152, 1, 2
assign g_t151, t943
reshape t944, g_t151, 512,768
assign g_t150, t944
assign g_t149, g_t150
sum t945, g_t150, 0
reshape t946, t945, 768
assign g_h4_attn_wk_b, t946
transpose t947, h4_attn_wk_w
matmul t948, g_t149, t947
add g_t143, g_t143, t948
transpose t949, t143
matmul t950, t949, g_t149
assign g_h4_attn_wk_w, t950
reshape t951, g_t148, 1,12,512,64
assign g_t147, t951
transpose t952, g_t147, 1, 2
assign g_t146, t952
reshape t953, g_t146, 512,768
assign g_t145, t953
assign g_t144, g_t145
sum t954, g_t145, 0
reshape t955, t954, 768
assign g_h4_attn_wq_b, t955
transpose t956, h4_attn_wq_w
matmul t957, g_t144, t956
add g_t143, g_t143, t957
transpose t958, t143
matmul t959, t958, g_t144
assign g_h4_attn_wq_w, t959
layernorm_grad t960, t961, t962, g_t143, t142, h4_ln1_g
add g_t142, g_t142, t960
assign g_h4_ln1_g, t961
assign g_h4_ln1_b, t962
assign g_t135, g_t142
assign g_t141, g_t142
assign g_t140, g_t141
sum t963, g_t141, 0
reshape t964, t963, 768
assign g_h3_mlp_proj_b, t964
transpose t965, h3_mlp_proj_w
matmul t966, g_t140, t965
assign g_t139, t966
transpose t967, t139
matmul t968, t967, g_t140
assign g_h3_mlp_proj_w, t968
gelu_grad t969, g_t139, t138
assign g_t138, t969
assign g_t137, g_t138
sum t970, g_t138, 0
reshape t971, t970, 3072
assign g_h3_mlp_fc_b, t971
transpose t972, h3_mlp_fc_w
matmul t973, g_t137, t972
assign g_t136, t973
transpose t974, t136
matmul t975, t974, g_t137
assign g_h3_mlp_fc_w, t975
layernorm_grad t976, t977, t978, g_t136, t135, h3_ln2_g
add g_t135, g_t135, t976
assign g_h3_ln2_g, t977
assign g_h3_ln2_b, t978
assign g_t107, g_t135
assign g_t134, g_t135
assign g_t133, g_t134
sum t979, g_t134, 0
reshape t980, t979, 768
assign g_h3_attn_wo_b, t980
transpose t981, h3_attn_wo_w
matmul t982, g_t133, t981
assign g_t132, t982
transpose t983, t132
matmul t984, t983, g_t133
assign g_h3_attn_wo_w, t984
reshape t985, g_t132, 1,512,12,64
assign g_t131, t985
transpose t986, g_t131, 1, 2
assign g_t130, t986
reshape t987, g_t130, 12,512,64
assign g_t129, t987
transpose t988, t123
matmul t989, g_t129, t988
assign g_t128, t989
transpose t990, t128
matmul t991, t990, g_t129
assign g_t123, t991
reshape t992, g_t123, 1,12,512,64
assign g_t122, t992
transpose t993, g_t122, 1, 2
assign g_t121, t993
reshape t994, g_t121, 512,768
assign g_t120, t994
assign g_t119, g_t120
sum t995, g_t120, 0
reshape t996, t995, 768
assign g_h3_attn_wv_b, t996
transpose t997, h3_attn_wv_w
matmul t998, g_t119, t997
assign g_t108, t998
transpose t999, t108
matmul t1000, t999, g_t119
assign g_h3_attn_wv_w, t1000
softmax_grad t1001, g_t128, t128
assign g_t127, t1001
assign g_t126, g_t127
mul t1002, g_t126, 0.125
assign g_t125, t1002
transpose t1003, t124
matmul t1004, g_t125, t1003
assign g_t113, t1004
transpose t1005, t113
matmul t1006, t1005, g_t125
assign g_t124, t1006
transpose t1007, g_t124, 1, 2
assign g_t118, t1007
reshape t1008, g_t118, 1,12,512,64
assign g_t117, t1008
transpose t1009, g_t117, 1, 2
assign g_t116, t1009
reshape t1010, g_t116, 512,768
assign g_t115, t1010
assign g_t114, g_t115
sum t1011, g_t115, 0
reshape t1012, t1011, 768
assign g_h3_attn_wk_b, t1012
transpose t1013, h3_attn_wk_w
matmul t1014, g_t114, t1013
add g_t108, g_t108, t1014
transpose t1015, t108
matmul t1016, t1015, g_t114
assign g_h3_attn_wk_w, t1016
reshape t1017, g_t113, 1,12,512,64
assign g_t112, t1017
transpose t1018, g_t112, 1, 2
assign g_t111, t1018
reshape t1019, g_t111, 512,768
assign g_t110, t1019
assign g_t109, g_t110
sum t1020, g_t110, 0
reshape t1021, t1020, 768
assign g_h3_attn_wq_b, t1021
transpose t1022, h3_attn_wq_w
matmul t1023, g_t109, t1022
add g_t108, g_t108, t1023
transpose t1024, t108
matmul t1025, t1024, g_t109
assign g_h3_attn_wq_w, t1025
layernorm_grad t1026, t1027, t1028, g_t108, t107, h3_ln1_g
add g_t107, g_t107, t1026
assign g_h3_ln1_g, t1027
assign g_h3_ln1_b, t1028
assign g_t100, g_t107
assign g_t106, g_t107
assign g_t105, g_t106
sum t1029, g_t106, 0
reshape t1030, t1029, 768
assign g_h2_mlp_proj_b, t1030
transpose t1031, h2_mlp_proj_w
matmul t1032, g_t105, t1031
assign g_t104, t1032
transpose t1033, t104
matmul t1034, t1033, g_t105
assign g_h2_mlp_proj_w, t1034
gelu_grad t1035, g_t104, t103
assign g_t103, t1035
assign g_t102, g_t103
sum t1036, g_t103, 0
reshape t1037, t1036, 3072
assign g_h2_mlp_fc_b, t1037
transpose t1038, h2_mlp_fc_w
matmul t1039, g_t102, t1038
assign g_t101, t1039
transpose t1040, t101
matmul t1041, t1040, g_t102
assign g_h2_mlp_fc_w, t1041
layernorm_grad t1042, t1043, t1044, g_t101, t100, h2_ln2_g
add g_t100, g_t100, t1042
assign g_h2_ln2_g, t1043
assign g_h2_ln2_b, t1044
assign g_t72, g_t100
assign g_t99, g_t100
assign g_t98, g_t99
sum t1045, g_t99, 0
reshape t1046, t1045, 768
assign g_h2_attn_wo_b, t1046
transpose t1047, h2_attn_wo_w
matmul t1048, g_t98, t1047
assign g_t97, t1048
transpose t1049, t97
matmul t1050, t1049, g_t98
assign g_h2_attn_wo_w, t1050
reshape t1051, g_t97, 1,512,12,64
assign g_t96, t1051
transpose t1052, g_t96, 1, 2
assign g_t95, t1052
reshape t1053, g_t95, 12,512,64
assign g_t94, t1053
transpose t1054, t88
matmul t1055, g_t94, t1054
assign g_t93, t1055
transpose t1056, t93
matmul t1057, t1056, g_t94
assign g_t88, t1057
reshape t1058, g_t88, 1,12,512,64
assign g_t87, t1058
transpose t1059, g_t87, 1, 2
assign g_t86, t1059
reshape t1060, g_t86, 512,768
assign g_t85, t1060
assign g_t84, g_t85
sum t1061, g_t85, 0
reshape t1062, t1061, 768
assign g_h2_attn_wv_b, t1062
transpose t1063, h2_attn_wv_w
matmul t1064, g_t84, t1063
assign g_t73, t1064
transpose t1065, t73
matmul t1066, t1065, g_t84
assign g_h2_attn_wv_w, t1066
softmax_grad t1067, g_t93, t93
assign g_t92, t1067
assign g_t91, g_t92
mul t1068, g_t91, 0.125
assign g_t90, t1068
transpose t1069, t89
matmul t1070, g_t90, t1069
assign g_t78, t1070
transpose t1071, t78
matmul t1072, t1071, g_t90
assign g_t89, t1072
transpose t1073, g_t89, 1, 2
assign g_t83, t1073
reshape t1074, g_t83, 1,12,512,64
assign g_t82, t1074
transpose t1075, g_t82, 1, 2
assign g_t81, t1075
reshape t1076, g_t81, 512,768
assign g_t80, t1076
assign g_t79, g_t80
sum t1077, g_t80, 0
reshape t1078, t1077, 768
assign g_h2_attn_wk_b, t1078
transpose t1079, h2_attn_wk_w
matmul t1080, g_t79, t1079
add g_t73, g_t73, t1080
transpose t1081, t73
matmul t1082, t1081, g_t79
assign g_h2_attn_wk_w, t1082
reshape t1083, g_t78, 1,12,512,64
assign g_t77, t1083
transpose t1084, g_t77, 1, 2
assign g_t76, t1084
reshape t1085, g_t76, 512,768
assign g_t75, t1085
assign g_t74, g_t75
sum t1086, g_t75, 0
reshape t1087, t1086, 768
assign g_h2_attn_wq_b, t1087
transpose t1088, h2_attn_wq_w
matmul t1089, g_t74, t1088
add g_t73, g_t73, t1089
transpose t1090, t73
matmul t1091, t1090, g_t74
assign g_h2_attn_wq_w, t1091
layernorm_grad t1092, t1093, t1094, g_t73, t72, h2_ln1_g
add g_t72, g_t72, t1092
assign g_h2_ln1_g, t1093
assign g_h2_ln1_b, t1094
assign g_t65, g_t72
assign g_t71, g_t72
assign g_t70, g_t71
sum t1095, g_t71, 0
reshape t1096, t1095, 768
assign g_h1_mlp_proj_b, t1096
transpose t1097, h1_mlp_proj_w
matmul t1098, g_t70, t1097
assign g_t69, t1098
transpose t1099, t69
matmul t1100, t1099, g_t70
assign g_h1_mlp_proj_w, t1100
gelu_grad t1101, g_t69, t68
assign g_t68, t1101
assign g_t67, g_t68
sum t1102, g_t68, 0
reshape t1103, t1102, 3072
assign g_h1_mlp_fc_b, t1103
transpose t1104, h1_mlp_fc_w
matmul t1105, g_t67, t1104
assign g_t66, t1105
transpose t1106, t66
matmul t1107, t1106, g_t67
assign g_h1_mlp_fc_w, t1107
layernorm_grad t1108, t1109, t1110, g_t66, t65, h1_ln2_g
add g_t65, g_t65, t1108
assign g_h1_ln2_g, t1109
assign g_h1_ln2_b, t1110
assign g_t37, g_t65
assign g_t64, g_t65
assign g_t63, g_t64
sum t1111, g_t64, 0
reshape t1112, t1111, 768
assign g_h1_attn_wo_b, t1112
transpose t1113, h1_attn_wo_w
matmul t1114, g_t63, t1113
assign g_t62, t1114
transpose t1115, t62
matmul t1116, t1115, g_t63
assign g_h1_attn_wo_w, t1116
reshape t1117, g_t62, 1,512,12,64
assign g_t61, t1117
transpose t1118, g_t61, 1, 2
assign g_t60, t1118
reshape t1119, g_t60, 12,512,64
assign g_t59, t1119
transpose t1120, t53
matmul t1121, g_t59, t1120
assign g_t58, t1121
transpose t1122, t58
matmul t1123, t1122, g_t59
assign g_t53, t1123
reshape t1124, g_t53, 1,12,512,64
assign g_t52, t1124
transpose t1125, g_t52, 1, 2
assign g_t51, t1125
reshape t1126, g_t51, 512,768
assign g_t50, t1126
assign g_t49, g_t50
sum t1127, g_t50, 0
reshape t1128, t1127, 768
assign g_h1_attn_wv_b, t1128
transpose t1129, h1_attn_wv_w
matmul t1130, g_t49, t1129
assign g_t38, t1130
transpose t1131, t38
matmul t1132, t1131, g_t49
assign g_h1_attn_wv_w, t1132
softmax_grad t1133, g_t58, t58
assign g_t57, t1133
assign g_t56, g_t57
mul t1134, g_t56, 0.125
assign g_t55, t1134
transpose t1135, t54
matmul t1136, g_t55, t1135
assign g_t43, t1136
transpose t1137, t43
matmul t1138, t1137, g_t55
assign g_t54, t1138
transpose t1139, g_t54, 1, 2
assign g_t48, t1139
reshape t1140, g_t48, 1,12,512,64
assign g_t47, t1140
transpose t1141, g_t47, 1, 2
assign g_t46, t1141
reshape t1142, g_t46, 512,768
assign g_t45, t1142
assign g_t44, g_t45
sum t1143, g_t45, 0
reshape t1144, t1143, 768
assign g_h1_attn_wk_b, t1144
transpose t1145, h1_attn_wk_w
matmul t1146, g_t44, t1145
add g_t38, g_t38, t1146
transpose t1147, t38
matmul t1148, t1147, g_t44
assign g_h1_attn_wk_w, t1148
reshape t1149, g_t43, 1,12,512,64
assign g_t42, t1149
transpose t1150, g_t42, 1, 2
assign g_t41, t1150
reshape t1151, g_t41, 512,768
assign g_t40, t1151
assign g_t39, g_t40
sum t1152, g_t40, 0
reshape t1153, t1152, 768
assign g_h1_attn_wq_b, t1153
transpose t1154, h1_attn_wq_w
matmul t1155, g_t39, t1154
add g_t38, g_t38, t1155
transpose t1156, t38
matmul t1157, t1156, g_t39
assign g_h1_attn_wq_w, t1157
layernorm_grad t1158, t1159, t1160, g_t38, t37, h1_ln1_g
add g_t37, g_t37, t1158
assign g_h1_ln1_g, t1159
assign g_h1_ln1_b, t1160
assign g_t30, g_t37
assign g_t36, g_t37
assign g_t35, g_t36
sum t1161, g_t36, 0
reshape t1162, t1161, 768
assign g_h0_mlp_proj_b, t1162
transpose t1163, h0_mlp_proj_w
matmul t1164, g_t35, t1163
assign g_t34, t1164
transpose t1165, t34
matmul t1166, t1165, g_t35
assign g_h0_mlp_proj_w, t1166
gelu_grad t1167, g_t34, t33
assign g_t33, t1167
assign g_t32, g_t33
sum t1168, g_t33, 0
reshape t1169, t1168, 3072
assign g_h0_mlp_fc_b, t1169
transpose t1170, h0_mlp_fc_w
matmul t1171, g_t32, t1170
assign g_t31, t1171
transpose t1172, t31
matmul t1173, t1172, g_t32
assign g_h0_mlp_fc_w, t1173
layernorm_grad t1174, t1175, t1176, g_t31, t30, h0_ln2_g
add g_t30, g_t30, t1174
assign g_h0_ln2_g, t1175
assign g_h0_ln2_b, t1176
assign g_t2, g_t30
assign g_t29, g_t30
assign g_t28, g_t29
sum t1177, g_t29, 0
reshape t1178, t1177, 768
assign g_h0_attn_wo_b, t1178
transpose t1179, h0_attn_wo_w
matmul t1180, g_t28, t1179
assign g_t27, t1180
transpose t1181, t27
matmul t1182, t1181, g_t28
assign g_h0_attn_wo_w, t1182
reshape t1183, g_t27, 1,512,12,64
assign g_t26, t1183
transpose t1184, g_t26, 1, 2
assign g_t25, t1184
reshape t1185, g_t25, 12,512,64
assign g_t24, t1185
transpose t1186, t18
matmul t1187, g_t24, t1186
assign g_t23, t1187
transpose t1188, t23
matmul t1189, t1188, g_t24
assign g_t18, t1189
reshape t1190, g_t18, 1,12,512,64
assign g_t17, t1190
transpose t1191, g_t17, 1, 2
assign g_t16, t1191
reshape t1192, g_t16, 512,768
assign g_t15, t1192
assign g_t14, g_t15
sum t1193, g_t15, 0
reshape t1194, t1193, 768
assign g_h0_attn_wv_b, t1194
transpose t1195, h0_attn_wv_w
matmul t1196, g_t14, t1195
assign g_t3, t1196
transpose t1197, t3
matmul t1198, t1197, g_t14
assign g_h0_attn_wv_w, t1198
softmax_grad t1199, g_t23, t23
assign g_t22, t1199
assign g_t21, g_t22
mul t1200, g_t21, 0.125
assign g_t20, t1200
transpose t1201, t19
matmul t1202, g_t20, t1201
assign g_t8, t1202
transpose t1203, t8
matmul t1204, t1203, g_t20
assign g_t19, t1204
transpose t1205, g_t19, 1, 2
assign g_t13, t1205
reshape t1206, g_t13, 1,12,512,64
assign g_t12, t1206
transpose t1207, g_t12, 1, 2
assign g_t11, t1207
reshape t1208, g_t11, 512,768
assign g_t10, t1208
assign g_t9, g_t10
sum t1209, g_t10, 0
reshape t1210, t1209, 768
assign g_h0_attn_wk_b, t1210
transpose t1211, h0_attn_wk_w
matmul t1212, g_t9, t1211
add g_t3, g_t3, t1212
transpose t1213, t3
matmul t1214, t1213, g_t9
assign g_h0_attn_wk_w, t1214
reshape t1215, g_t8, 1,12,512,64
assign g_t7, t1215
transpose t1216, g_t7, 1, 2
assign g_t6, t1216
reshape t1217, g_t6, 512,768
assign g_t5, t1217
assign g_t4, g_t5
sum t1218, g_t5, 0
reshape t1219, t1218, 768
assign g_h0_attn_wq_b, t1219
transpose t1220, h0_attn_wq_w
matmul t1221, g_t4, t1220
add g_t3, g_t3, t1221
transpose t1222, t3
matmul t1223, t1222, g_t4
assign g_h0_attn_wq_w, t1223
layernorm_grad t1224, t1225, t1226, g_t3, t2, h0_ln1_g
add g_t2, g_t2, t1224
assign g_h0_ln1_g, t1225
assign g_h0_ln1_b, t1226
assign g_t0, g_t2
assign g_t1, g_t2
embedding_grad t1227, g_t1, pos
assign g_wpe, t1227
embedding_grad t1228, g_t0, tok
add g_wte, g_wte, t1228