# === FORWARD PASS ===
embedding t0, wte, tok
embedding t1, wpe, pos
add t2, t0, t1
layernorm t3, t2, h0_ln1_g, h0_ln1_b
matmul t4, t3, h0_attn_wq_w
add t5, t4, h0_attn_wq_b
reshape t6, t5, 4,128,4,32
transpose t7, t6, 1, 2
reshape t8, t7, 16,128,32
matmul t9, t3, h0_attn_wk_w
add t10, t9, h0_attn_wk_b
reshape t11, t10, 4,128,4,32
transpose t12, t11, 1, 2
reshape t13, t12, 16,128,32
transpose t19, t13, 1, 2
matmul t20, t8, t19
mul t21, t20, 0.17677669529663687
add t22, t21, mask
softmax t23, t22
matmul t14, t3, h0_attn_wv_w
add t15, t14, h0_attn_wv_b
reshape t16, t15, 4,128,4,32
transpose t17, t16, 1, 2
reshape t18, t17, 16,128,32
matmul t24, t23, t18
reshape t25, t24, 4,4,128,32
transpose t26, t25, 1, 2
reshape t27, t26, 512,128
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
reshape t41, t40, 4,128,4,32
transpose t42, t41, 1, 2
reshape t43, t42, 16,128,32
matmul t44, t38, h1_attn_wk_w
add t45, t44, h1_attn_wk_b
reshape t46, t45, 4,128,4,32
transpose t47, t46, 1, 2
reshape t48, t47, 16,128,32
transpose t54, t48, 1, 2
matmul t55, t43, t54
mul t56, t55, 0.17677669529663687
add t57, t56, mask
softmax t58, t57
matmul t49, t38, h1_attn_wv_w
add t50, t49, h1_attn_wv_b
reshape t51, t50, 4,128,4,32
transpose t52, t51, 1, 2
reshape t53, t52, 16,128,32
matmul t59, t58, t53
reshape t60, t59, 4,4,128,32
transpose t61, t60, 1, 2
reshape t62, t61, 512,128
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
reshape t76, t75, 4,128,4,32
transpose t77, t76, 1, 2
reshape t78, t77, 16,128,32
matmul t79, t73, h2_attn_wk_w
add t80, t79, h2_attn_wk_b
reshape t81, t80, 4,128,4,32
transpose t82, t81, 1, 2
reshape t83, t82, 16,128,32
transpose t89, t83, 1, 2
matmul t90, t78, t89
mul t91, t90, 0.17677669529663687
add t92, t91, mask
softmax t93, t92
matmul t84, t73, h2_attn_wv_w
add t85, t84, h2_attn_wv_b
reshape t86, t85, 4,128,4,32
transpose t87, t86, 1, 2
reshape t88, t87, 16,128,32
matmul t94, t93, t88
reshape t95, t94, 4,4,128,32
transpose t96, t95, 1, 2
reshape t97, t96, 512,128
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
reshape t111, t110, 4,128,4,32
transpose t112, t111, 1, 2
reshape t113, t112, 16,128,32
matmul t114, t108, h3_attn_wk_w
add t115, t114, h3_attn_wk_b
reshape t116, t115, 4,128,4,32
transpose t117, t116, 1, 2
reshape t118, t117, 16,128,32
transpose t124, t118, 1, 2
matmul t125, t113, t124
mul t126, t125, 0.17677669529663687
add t127, t126, mask
softmax t128, t127
matmul t119, t108, h3_attn_wv_w
add t120, t119, h3_attn_wv_b
reshape t121, t120, 4,128,4,32
transpose t122, t121, 1, 2
reshape t123, t122, 16,128,32
matmul t129, t128, t123
reshape t130, t129, 4,4,128,32
transpose t131, t130, 1, 2
reshape t132, t131, 512,128
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
layernorm t143, t142, ln_f_g, ln_f_b
transpose t144, wte, 0, 1
matmul t145, t143, t144
cross_entropy t146, g_t145, t145, targets

# === BACKWARD PASS ===
transpose t147, t144
matmul t148, g_t145, t147
assign g_t143, t148
transpose t149, t143
matmul t150, t149, g_t145
assign g_t144, t150
transpose t151, g_t144, 0, 1
assign g_wte, t151
layernorm_grad t152, t153, t154, g_t143, t142, ln_f_g
assign g_t142, t152
assign g_ln_f_g, t153
assign g_ln_f_b, t154
assign g_t135, g_t142
assign g_t141, g_t142
assign g_t140, g_t141
sum t155, g_t141, 0
reshape t156, t155, 128
assign g_h3_mlp_proj_b, t156
transpose t157, h3_mlp_proj_w
matmul t158, g_t140, t157
assign g_t139, t158
transpose t159, t139
matmul t160, t159, g_t140
assign g_h3_mlp_proj_w, t160
gelu_grad t161, g_t139, t138
assign g_t138, t161
assign g_t137, g_t138
sum t162, g_t138, 0
reshape t163, t162, 512
assign g_h3_mlp_fc_b, t163
transpose t164, h3_mlp_fc_w
matmul t165, g_t137, t164
assign g_t136, t165
transpose t166, t136
matmul t167, t166, g_t137
assign g_h3_mlp_fc_w, t167
layernorm_grad t168, t169, t170, g_t136, t135, h3_ln2_g
add g_t135, g_t135, t168
assign g_h3_ln2_g, t169
assign g_h3_ln2_b, t170
assign g_t107, g_t135
assign g_t134, g_t135
assign g_t133, g_t134
sum t171, g_t134, 0
reshape t172, t171, 128
assign g_h3_attn_wo_b, t172
transpose t173, h3_attn_wo_w
matmul t174, g_t133, t173
assign g_t132, t174
transpose t175, t132
matmul t176, t175, g_t133
assign g_h3_attn_wo_w, t176
reshape t177, g_t132, 4,128,4,32
assign g_t131, t177
transpose t178, g_t131, 1, 2
assign g_t130, t178
reshape t179, g_t130, 16,128,32
assign g_t129, t179
transpose t180, t123
matmul t181, g_t129, t180
assign g_t128, t181
transpose t182, t128
matmul t183, t182, g_t129
assign g_t123, t183
reshape t184, g_t123, 4,4,128,32
assign g_t122, t184
transpose t185, g_t122, 1, 2
assign g_t121, t185
reshape t186, g_t121, 512,128
assign g_t120, t186
assign g_t119, g_t120
sum t187, g_t120, 0
reshape t188, t187, 128
assign g_h3_attn_wv_b, t188
transpose t189, h3_attn_wv_w
matmul t190, g_t119, t189
assign g_t108, t190
transpose t191, t108
matmul t192, t191, g_t119
assign g_h3_attn_wv_w, t192
softmax_grad t193, g_t128, t128
assign g_t127, t193
assign g_t126, g_t127
mul t194, g_t126, 0.17677669529663687
assign g_t125, t194
transpose t195, t124
matmul t196, g_t125, t195
assign g_t113, t196
transpose t197, t113
matmul t198, t197, g_t125
assign g_t124, t198
transpose t199, g_t124, 1, 2
assign g_t118, t199
reshape t200, g_t118, 4,4,128,32
assign g_t117, t200
transpose t201, g_t117, 1, 2
assign g_t116, t201
reshape t202, g_t116, 512,128
assign g_t115, t202
assign g_t114, g_t115
sum t203, g_t115, 0
reshape t204, t203, 128
assign g_h3_attn_wk_b, t204
transpose t205, h3_attn_wk_w
matmul t206, g_t114, t205
add g_t108, g_t108, t206
transpose t207, t108
matmul t208, t207, g_t114
assign g_h3_attn_wk_w, t208
reshape t209, g_t113, 4,4,128,32
assign g_t112, t209
transpose t210, g_t112, 1, 2
assign g_t111, t210
reshape t211, g_t111, 512,128
assign g_t110, t211
assign g_t109, g_t110
sum t212, g_t110, 0
reshape t213, t212, 128
assign g_h3_attn_wq_b, t213
transpose t214, h3_attn_wq_w
matmul t215, g_t109, t214
add g_t108, g_t108, t215
transpose t216, t108
matmul t217, t216, g_t109
assign g_h3_attn_wq_w, t217
layernorm_grad t218, t219, t220, g_t108, t107, h3_ln1_g
add g_t107, g_t107, t218
assign g_h3_ln1_g, t219
assign g_h3_ln1_b, t220
assign g_t100, g_t107
assign g_t106, g_t107
assign g_t105, g_t106
sum t221, g_t106, 0
reshape t222, t221, 128
assign g_h2_mlp_proj_b, t222
transpose t223, h2_mlp_proj_w
matmul t224, g_t105, t223
assign g_t104, t224
transpose t225, t104
matmul t226, t225, g_t105
assign g_h2_mlp_proj_w, t226
gelu_grad t227, g_t104, t103
assign g_t103, t227
assign g_t102, g_t103
sum t228, g_t103, 0
reshape t229, t228, 512
assign g_h2_mlp_fc_b, t229
transpose t230, h2_mlp_fc_w
matmul t231, g_t102, t230
assign g_t101, t231
transpose t232, t101
matmul t233, t232, g_t102
assign g_h2_mlp_fc_w, t233
layernorm_grad t234, t235, t236, g_t101, t100, h2_ln2_g
add g_t100, g_t100, t234
assign g_h2_ln2_g, t235
assign g_h2_ln2_b, t236
assign g_t72, g_t100
assign g_t99, g_t100
assign g_t98, g_t99
sum t237, g_t99, 0
reshape t238, t237, 128
assign g_h2_attn_wo_b, t238
transpose t239, h2_attn_wo_w
matmul t240, g_t98, t239
assign g_t97, t240
transpose t241, t97
matmul t242, t241, g_t98
assign g_h2_attn_wo_w, t242
reshape t243, g_t97, 4,128,4,32
assign g_t96, t243
transpose t244, g_t96, 1, 2
assign g_t95, t244
reshape t245, g_t95, 16,128,32
assign g_t94, t245
transpose t246, t88
matmul t247, g_t94, t246
assign g_t93, t247
transpose t248, t93
matmul t249, t248, g_t94
assign g_t88, t249
reshape t250, g_t88, 4,4,128,32
assign g_t87, t250
transpose t251, g_t87, 1, 2
assign g_t86, t251
reshape t252, g_t86, 512,128
assign g_t85, t252
assign g_t84, g_t85
sum t253, g_t85, 0
reshape t254, t253, 128
assign g_h2_attn_wv_b, t254
transpose t255, h2_attn_wv_w
matmul t256, g_t84, t255
assign g_t73, t256
transpose t257, t73
matmul t258, t257, g_t84
assign g_h2_attn_wv_w, t258
softmax_grad t259, g_t93, t93
assign g_t92, t259
assign g_t91, g_t92
mul t260, g_t91, 0.17677669529663687
assign g_t90, t260
transpose t261, t89
matmul t262, g_t90, t261
assign g_t78, t262
transpose t263, t78
matmul t264, t263, g_t90
assign g_t89, t264
transpose t265, g_t89, 1, 2
assign g_t83, t265
reshape t266, g_t83, 4,4,128,32
assign g_t82, t266
transpose t267, g_t82, 1, 2
assign g_t81, t267
reshape t268, g_t81, 512,128
assign g_t80, t268
assign g_t79, g_t80
sum t269, g_t80, 0
reshape t270, t269, 128
assign g_h2_attn_wk_b, t270
transpose t271, h2_attn_wk_w
matmul t272, g_t79, t271
add g_t73, g_t73, t272
transpose t273, t73
matmul t274, t273, g_t79
assign g_h2_attn_wk_w, t274
reshape t275, g_t78, 4,4,128,32
assign g_t77, t275
transpose t276, g_t77, 1, 2
assign g_t76, t276
reshape t277, g_t76, 512,128
assign g_t75, t277
assign g_t74, g_t75
sum t278, g_t75, 0
reshape t279, t278, 128
assign g_h2_attn_wq_b, t279
transpose t280, h2_attn_wq_w
matmul t281, g_t74, t280
add g_t73, g_t73, t281
transpose t282, t73
matmul t283, t282, g_t74
assign g_h2_attn_wq_w, t283
layernorm_grad t284, t285, t286, g_t73, t72, h2_ln1_g
add g_t72, g_t72, t284
assign g_h2_ln1_g, t285
assign g_h2_ln1_b, t286
assign g_t65, g_t72
assign g_t71, g_t72
assign g_t70, g_t71
sum t287, g_t71, 0
reshape t288, t287, 128
assign g_h1_mlp_proj_b, t288
transpose t289, h1_mlp_proj_w
matmul t290, g_t70, t289
assign g_t69, t290
transpose t291, t69
matmul t292, t291, g_t70
assign g_h1_mlp_proj_w, t292
gelu_grad t293, g_t69, t68
assign g_t68, t293
assign g_t67, g_t68
sum t294, g_t68, 0
reshape t295, t294, 512
assign g_h1_mlp_fc_b, t295
transpose t296, h1_mlp_fc_w
matmul t297, g_t67, t296
assign g_t66, t297
transpose t298, t66
matmul t299, t298, g_t67
assign g_h1_mlp_fc_w, t299
layernorm_grad t300, t301, t302, g_t66, t65, h1_ln2_g
add g_t65, g_t65, t300
assign g_h1_ln2_g, t301
assign g_h1_ln2_b, t302
assign g_t37, g_t65
assign g_t64, g_t65
assign g_t63, g_t64
sum t303, g_t64, 0
reshape t304, t303, 128
assign g_h1_attn_wo_b, t304
transpose t305, h1_attn_wo_w
matmul t306, g_t63, t305
assign g_t62, t306
transpose t307, t62
matmul t308, t307, g_t63
assign g_h1_attn_wo_w, t308
reshape t309, g_t62, 4,128,4,32
assign g_t61, t309
transpose t310, g_t61, 1, 2
assign g_t60, t310
reshape t311, g_t60, 16,128,32
assign g_t59, t311
transpose t312, t53
matmul t313, g_t59, t312
assign g_t58, t313
transpose t314, t58
matmul t315, t314, g_t59
assign g_t53, t315
reshape t316, g_t53, 4,4,128,32
assign g_t52, t316
transpose t317, g_t52, 1, 2
assign g_t51, t317
reshape t318, g_t51, 512,128
assign g_t50, t318
assign g_t49, g_t50
sum t319, g_t50, 0
reshape t320, t319, 128
assign g_h1_attn_wv_b, t320
transpose t321, h1_attn_wv_w
matmul t322, g_t49, t321
assign g_t38, t322
transpose t323, t38
matmul t324, t323, g_t49
assign g_h1_attn_wv_w, t324
softmax_grad t325, g_t58, t58
assign g_t57, t325
assign g_t56, g_t57
mul t326, g_t56, 0.17677669529663687
assign g_t55, t326
transpose t327, t54
matmul t328, g_t55, t327
assign g_t43, t328
transpose t329, t43
matmul t330, t329, g_t55
assign g_t54, t330
transpose t331, g_t54, 1, 2
assign g_t48, t331
reshape t332, g_t48, 4,4,128,32
assign g_t47, t332
transpose t333, g_t47, 1, 2
assign g_t46, t333
reshape t334, g_t46, 512,128
assign g_t45, t334
assign g_t44, g_t45
sum t335, g_t45, 0
reshape t336, t335, 128
assign g_h1_attn_wk_b, t336
transpose t337, h1_attn_wk_w
matmul t338, g_t44, t337
add g_t38, g_t38, t338
transpose t339, t38
matmul t340, t339, g_t44
assign g_h1_attn_wk_w, t340
reshape t341, g_t43, 4,4,128,32
assign g_t42, t341
transpose t342, g_t42, 1, 2
assign g_t41, t342
reshape t343, g_t41, 512,128
assign g_t40, t343
assign g_t39, g_t40
sum t344, g_t40, 0
reshape t345, t344, 128
assign g_h1_attn_wq_b, t345
transpose t346, h1_attn_wq_w
matmul t347, g_t39, t346
add g_t38, g_t38, t347
transpose t348, t38
matmul t349, t348, g_t39
assign g_h1_attn_wq_w, t349
layernorm_grad t350, t351, t352, g_t38, t37, h1_ln1_g
add g_t37, g_t37, t350
assign g_h1_ln1_g, t351
assign g_h1_ln1_b, t352
assign g_t30, g_t37
assign g_t36, g_t37
assign g_t35, g_t36
sum t353, g_t36, 0
reshape t354, t353, 128
assign g_h0_mlp_proj_b, t354
transpose t355, h0_mlp_proj_w
matmul t356, g_t35, t355
assign g_t34, t356
transpose t357, t34
matmul t358, t357, g_t35
assign g_h0_mlp_proj_w, t358
gelu_grad t359, g_t34, t33
assign g_t33, t359
assign g_t32, g_t33
sum t360, g_t33, 0
reshape t361, t360, 512
assign g_h0_mlp_fc_b, t361
transpose t362, h0_mlp_fc_w
matmul t363, g_t32, t362
assign g_t31, t363
transpose t364, t31
matmul t365, t364, g_t32
assign g_h0_mlp_fc_w, t365
layernorm_grad t366, t367, t368, g_t31, t30, h0_ln2_g
add g_t30, g_t30, t366
assign g_h0_ln2_g, t367
assign g_h0_ln2_b, t368
assign g_t2, g_t30
assign g_t29, g_t30
assign g_t28, g_t29
sum t369, g_t29, 0
reshape t370, t369, 128
assign g_h0_attn_wo_b, t370
transpose t371, h0_attn_wo_w
matmul t372, g_t28, t371
assign g_t27, t372
transpose t373, t27
matmul t374, t373, g_t28
assign g_h0_attn_wo_w, t374
reshape t375, g_t27, 4,128,4,32
assign g_t26, t375
transpose t376, g_t26, 1, 2
assign g_t25, t376
reshape t377, g_t25, 16,128,32
assign g_t24, t377
transpose t378, t18
matmul t379, g_t24, t378
assign g_t23, t379
transpose t380, t23
matmul t381, t380, g_t24
assign g_t18, t381
reshape t382, g_t18, 4,4,128,32
assign g_t17, t382
transpose t383, g_t17, 1, 2
assign g_t16, t383
reshape t384, g_t16, 512,128
assign g_t15, t384
assign g_t14, g_t15
sum t385, g_t15, 0
reshape t386, t385, 128
assign g_h0_attn_wv_b, t386
transpose t387, h0_attn_wv_w
matmul t388, g_t14, t387
assign g_t3, t388
transpose t389, t3
matmul t390, t389, g_t14
assign g_h0_attn_wv_w, t390
softmax_grad t391, g_t23, t23
assign g_t22, t391
assign g_t21, g_t22
mul t392, g_t21, 0.17677669529663687
assign g_t20, t392
transpose t393, t19
matmul t394, g_t20, t393
assign g_t8, t394
transpose t395, t8
matmul t396, t395, g_t20
assign g_t19, t396
transpose t397, g_t19, 1, 2
assign g_t13, t397
reshape t398, g_t13, 4,4,128,32
assign g_t12, t398
transpose t399, g_t12, 1, 2
assign g_t11, t399
reshape t400, g_t11, 512,128
assign g_t10, t400
assign g_t9, g_t10
sum t401, g_t10, 0
reshape t402, t401, 128
assign g_h0_attn_wk_b, t402
transpose t403, h0_attn_wk_w
matmul t404, g_t9, t403
add g_t3, g_t3, t404
transpose t405, t3
matmul t406, t405, g_t9
assign g_h0_attn_wk_w, t406
reshape t407, g_t8, 4,4,128,32
assign g_t7, t407
transpose t408, g_t7, 1, 2
assign g_t6, t408
reshape t409, g_t6, 512,128
assign g_t5, t409
assign g_t4, g_t5
sum t410, g_t5, 0
reshape t411, t410, 128
assign g_h0_attn_wq_b, t411
transpose t412, h0_attn_wq_w
matmul t413, g_t4, t412
add g_t3, g_t3, t413
transpose t414, t3
matmul t415, t414, g_t4
assign g_h0_attn_wq_w, t415
layernorm_grad t416, t417, t418, g_t3, t2, h0_ln1_g
add g_t2, g_t2, t416
assign g_h0_ln1_g, t417
assign g_h0_ln1_b, t418
assign g_t0, g_t2
assign g_t1, g_t2
embedding_grad t419, g_t1, pos
assign g_wpe, t419
embedding_grad t420, g_t0, tok
add g_wte, g_wte, t420