template <
    typename T1,
    typename T2,
    typename FP,
    int SPLINE_DIM, int AnchorBlockSizeX,
    int AnchorBlockSizeY, int AnchorBlockSizeZ,
    int numAnchorBlockX,
    int numAnchorBlockY,
    int numAnchorBlockZ,
    typename LAMBDA,
    bool LINE,
    bool FACE,
    bool CUBE,
    int  LINEAR_BLOCK_SIZE,
    bool COARSEN,
    bool BORDER_INCLUSIVE,
    bool WORKFLOW,
    typename INTERP>
__forceinline__ __device__ void interpolate_stage_md(
    volatile T1 s_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
    [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
    [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
volatile T2 s_ectrl[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
 [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
 [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
    DIM3    data_size,
    LAMBDA xyzmap,
    int         unit,
    FP          eb_r,
    FP          ebx2,
    int         radius,
    INTERP cubic_interpolator,
    int NUM_ELE)
{
    // static_assert(COARSEN or (NUM_ELE <= BLOCK_DIM_SIZE), "block oversized");
    static_assert((LINE or FACE or CUBE) == true, "must be one hot");
    static_assert((LINE and FACE) == false, "must be only one hot (1)");
    static_assert((LINE and CUBE) == false, "must be only one hot (2)");
    static_assert((FACE and CUBE) == false, "must be only one hot (3)");

    auto run = [&](auto x, auto y, auto z) {

        if (xyz_predicate<SPLINE_DIM,
            AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
            numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, BORDER_INCLUSIVE>(x, y, z,data_size)) {
            T1 pred = 0;
            auto global_x = BIX * AnchorBlockSizeX * numAnchorBlockX + x;
            auto global_y = BIY * AnchorBlockSizeY * numAnchorBlockY + y;
            auto global_z = BIZ * AnchorBlockSizeZ * numAnchorBlockZ + z;  

           T1 tmp_z[4], tmp_y[4], tmp_x[4];
           int id_z[4], id_y[4], id_x[4];
           id_z[0] = (z - 3 * unit >= 0) ? z - 3 * unit : 0;
           id_z[1] = (z - unit >= 0) ? z - unit : 0;
           id_z[2] = (z + unit <= AnchorBlockSizeZ * numAnchorBlockZ) ? z + unit : 0;
           id_z[3] = (z + 3 * unit <= AnchorBlockSizeZ * numAnchorBlockZ) ? z + 3 * unit : 0;
           
           id_y[0] = (y - 3 * unit >= 0) ? y - 3 * unit : 0;
           id_y[1] = (y - unit >= 0) ? y - unit : 0;
           id_y[2] = (y + unit <= AnchorBlockSizeY * numAnchorBlockY) ? y + unit : 0;
           id_y[3] = (y + 3 * unit <= AnchorBlockSizeY * numAnchorBlockY) ? y + 3 * unit : 0;
           
           id_x[0] = (x - 3 * unit >= 0) ? x - 3 * unit : 0;
           id_x[1] = (x - unit >= 0) ? x - unit : 0;
           id_x[2] = (x + unit <= AnchorBlockSizeX * numAnchorBlockX) ? x + unit : 0;
           id_x[3] = (x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX) ? x + 3 * unit : 0;
           
            if CONSTEXPR (LINE) {
                bool I_Y = (y % (2*unit) )> 0; 
                bool I_Z = (z % (2*unit) )> 0; 

                pred = 0;
                auto input_x = x;
                auto input_BI = BIX;
                auto input_GD = GDX;
                auto input_gx = global_x;
                auto input_gs = data_size.x;

                auto right_bound = AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1);
                auto x_size = AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1);
                auto y_size = AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2);
                auto z_size = AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3);
                
                if (I_Z){
                    input_x = z;
                    input_BI = BIZ;
                    input_GD = GDZ;
                    input_gx = global_z;
                    input_gs = data_size.z;
                    right_bound = AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3);
                }
                else if (I_Y){
                    input_x = y;
                    input_BI = BIY;
                    input_GD = GDY;
                    input_gx = global_y;
                    input_gs = data_size.y;
                    right_bound = AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2);
                }
                
                int id_[4], s_id[4];
                id_[0] =  input_x - 3 * unit;
                id_[0] =  id_[0] >= 0 ? id_[0] : 0;
            
                id_[1] = input_x - unit;
                id_[1] = id_[1] >= 0 ? id_[1] : 0;
            
                id_[2] = input_x + unit;
                id_[2] = id_[2] < right_bound ? id_[2] : 0;
                
                id_[3] = input_x + 3 * unit;
                id_[3] = id_[3] < right_bound ? id_[3] : 0;
                
                s_id[0] = x_size * y_size * z + x_size * y + id_[0];
                s_id[1] = x_size * y_size * z + x_size * y + id_[1];
                s_id[2] = x_size * y_size * z + x_size * y + id_[2];
                s_id[3] = x_size * y_size * z + x_size * y + id_[3];
                if (I_Z){
                s_id[0] = x_size * y_size * id_[0] + x_size * y + x;
                s_id[1] = x_size * y_size * id_[1] + x_size * y + x;
                s_id[2] = x_size * y_size * id_[2] + x_size * y + x;
                s_id[3] = x_size * y_size * id_[3] + x_size * y + x;
                }
                else if (I_Y){
                    s_id[0] = x_size * y_size * z + x_size * id_[0] + x;
                    s_id[1] = x_size * y_size * z + x_size * id_[1] + x;
                    s_id[2] = x_size * y_size * z + x_size * id_[2] + x;
                    s_id[3] = x_size * y_size * z + x_size * id_[3] + x;
                }

                T1 tmp_[4];
            
                bool case1 = (input_BI != input_GD - 1);
                bool case2 = (input_x >= 3 * unit);
                bool case3 = (input_x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX);
                bool case4 = (input_gx + 3 * unit < input_gs);
                bool case5 = (input_gx + unit < input_gs);
                
                
                // 预加载 shared memory 数据到寄存器
                T1 tmp0 = *((T1*)s_data + s_id[0]); 
                T1 tmp1 = *((T1*)s_data + s_id[1]); 
                T1 tmp2 = *((T1*)s_data + s_id[2]); 
                T1 tmp3 = *((T1*)s_data + s_id[3]); 
    
                // 初始预测值
                pred = tmp1;
    
                // 计算不同 case 对应的 pred
                if ( (case1 && case2 && case3) || (!case1 && case2 && case3 && case4)) {
                    pred = cubic_interpolator(tmp0, tmp1, tmp2, tmp3);
                    
                }
                else if ((case1 && case2 && !case3) || ( !case1 && case2 && !(case3 && case4) && case5)) {
                    pred = (-tmp0 + 6 * tmp1 + 3 * tmp2) / 8;
                }
                else if ((case1 && !case2 && case3) || (!case1 && !case2 && case3 && case4 )){
                    pred = (3 * tmp1 + 6 * tmp2 - tmp3) / 8;   
                }
                else if ((case1 && !case2 && !case3) || (!case1 && !case2 && !(case3 && case4) && case5)) {
                    pred = (tmp1 + tmp2) / 2;
                }

            }
            auto get_interp_order = [&](auto x, auto BI, auto GD, auto gx, auto gs){
                int b = (x >= 3 * unit) ? 3 : 1;
                int f = ((x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX) && ((BI != GD - 1) || (gx + 3 * unit < gs))) ? 3 :
                (((BI != GD - 1) || (gx + unit < gs)) ? 1 : 0);

                return (b == 3) ? ((f == 3) ? 4 : ((f == 1) ? 3 : 0)) 
                                : ((f == 3) ? 2 : ((f == 1) ? 1 : 0));
            };
            if CONSTEXPR (FACE) {  //
               // if(BIX == 5 and BIY == 22 and BIZ == 6 and unit==1 and x==29 and y==7 and z==0){
               //     printf("%.2e %.2e %.2e %.2e\n",s_data[z ][y- 3*unit][x],s_data[z ][y- unit][x],s_data[z ][y+ unit][x]);
              //  }

                bool I_YZ = (x % (2*unit) ) == 0;
                bool I_XZ = (y % (2*unit ) )== 0;

                //if(BIX == 10 and BIY == 12 and BIZ == 0 and x==13 and y==6 and z==9)
               //     printf("face %d %d\n", I_YZ,I_XZ);
                int x_1,BI_1,GD_1,gx_1,gs_1;
                int x_2,BI_2,GD_2,gx_2,gs_2;
                int s_id_1[4], s_id_2[4];
                auto x_size = AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1);
                auto y_size = AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2);
                auto z_size = AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3);
                if (I_YZ){
                   
                 x_1 = z,BI_1 = BIZ, GD_1 = GDZ, gx_1 = global_z,gs_1 = data_size.z;
                 x_2 = y,BI_2 = BIY, GD_2 = GDY, gx_2 = global_y, gs_2 = data_size.y;
                 s_id_1[0] = x_size * y_size * id_z[0] + x_size * y + x;
                 s_id_1[1] = x_size * y_size * id_z[1] + x_size * y + x;
                 s_id_1[2] = x_size * y_size * id_z[2] + x_size * y + x;
                 s_id_1[3] = x_size * y_size * id_z[3] + x_size * y + x;
                 s_id_2[0] = x_size * y_size * z + x_size * id_y[0] + x;
                 s_id_2[1] = x_size * y_size * z + x_size * id_y[1] + x;
                 s_id_2[2] = x_size * y_size * z + x_size * id_y[2] + x;
                 s_id_2[3] = x_size * y_size * z + x_size * id_y[3] + x;
                 pred = s_data[id_z[1]][id_y[1]][x];

                }
                else if (I_XZ){
                    x_1 = z,BI_1 = BIZ, GD_1 = GDZ, gx_1 = global_z,gs_1 = data_size.z;
                    x_2 = x,BI_2 = BIX, GD_2 = GDX, gx_2 = global_x, gs_2 = data_size.x;
                    s_id_1[0] = x_size * y_size * id_z[0] + x_size * y + x;
                    s_id_1[1] = x_size * y_size * id_z[1] + x_size * y + x;
                    s_id_1[2] = x_size * y_size * id_z[2] + x_size * y + x;
                    s_id_1[3] = x_size * y_size * id_z[3] + x_size * y + x;
                    
                    s_id_2[0] = x_size * y_size * z + x_size * y + id_x[0];
                    s_id_2[1] = x_size * y_size * z + x_size * y + id_x[1];
                    s_id_2[2] = x_size * y_size * z + x_size * y + id_x[2];
                    s_id_2[3] = x_size * y_size * z + x_size * y + id_x[3];
                    pred = s_data[id_z[1]][y][id_x[1]];
                    
                }
                else{
                    x_1 = y,BI_1 = BIY, GD_1 = GDY, gx_1 = global_y, gs_1 = data_size.y;
                    x_2 = x,BI_2 = BIX, GD_2 = GDX, gx_2 = global_x, gs_2 = data_size.x;
                    s_id_1[0] = x_size * y_size * z + x_size * id_y[0] + x;
                    s_id_1[1] = x_size * y_size * z + x_size * id_y[1] + x;
                    s_id_1[2] = x_size * y_size * z + x_size * id_y[2] + x;
                    s_id_1[3] = x_size * y_size * z + x_size * id_y[3] + x;
                    s_id_2[0] = x_size * y_size * z + x_size * y + id_x[0];
                    s_id_2[1] = x_size * y_size * z + x_size * y + id_x[1];
                    s_id_2[2] = x_size * y_size * z + x_size * y + id_x[2];
                    s_id_2[3] = x_size * y_size * z + x_size * y + id_x[3];
                    pred = s_data[z][id_y[1]][id_x[1]];
                }

                    auto interp_1 = get_interp_order(x_1,BI_1,GD_1,gx_1,gs_1);
                    auto interp_2 = get_interp_order(x_2,BI_2,GD_2,gx_2,gs_2);

                    int case_num = interp_1 + interp_2 * 5;


                    if (interp_1 == 4 && interp_2 == 4) {
                        pred = (cubic_interpolator(*((T1*)s_data + s_id_1[0]), 
                        *((T1*)s_data + s_id_1[1]), 
                        *((T1*)s_data + s_id_1[2]), 
                        *((T1*)s_data + s_id_1[3])) +
                         cubic_interpolator(*((T1*)s_data + s_id_2[0]), 
                        *((T1*)s_data + s_id_2[1]), 
                        *((T1*)s_data + s_id_2[2]), 
                        *((T1*)s_data + s_id_2[3]))) / 2;
                    } else if (interp_1 != 4 && interp_2 == 4) {
                        pred = cubic_interpolator(*((T1*)s_data + s_id_2[0]), 
                        *((T1*)s_data + s_id_2[1]), 
                        *((T1*)s_data + s_id_2[2]), 
                        *((T1*)s_data + s_id_2[3]));
                    } else if (interp_1 == 4 && interp_2 != 4) {
                        pred = cubic_interpolator(*((T1*)s_data + s_id_1[0]), 
                        *((T1*)s_data + s_id_1[1]), 
                        *((T1*)s_data + s_id_1[2]), 
                        *((T1*)s_data + s_id_1[3]));
                    } else if (interp_1 == 3 && interp_2 == 3) {
                        pred = (-(*((T1*)s_data + s_id_2[0]))+6*(*((T1*)s_data + s_id_2[1])) + 3*(*((T1*)s_data + s_id_2[2]))) / 8;
                        pred += (-(*((T1*)s_data + s_id_1[0]))+6*(*((T1*)s_data + s_id_1[1])) + 3*(*((T1*)s_data + s_id_1[2]))) / 8;
                        pred /= 2;
                    } else if (interp_1 == 3 && interp_2 == 2) {
                        pred = (3*(*((T1*)s_data + s_id_2[1]))+6*(*((T1*)s_data + s_id_2[2])) - (*((T1*)s_data + s_id_2[3]))) / 8;
                        pred += (-(*((T1*)s_data + s_id_1[0]))+6*(*((T1*)s_data + s_id_1[1])) + 3*(*((T1*)s_data + s_id_1[2]))) / 8;
                        pred /= 2;
                    } else if (interp_1 == 3 && interp_2 < 2) {
                        pred = (-(*((T1*)s_data + s_id_1[0]))+6*(*((T1*)s_data + s_id_1[1])) + 3*(*((T1*)s_data + s_id_1[2]))) / 8;
                    } else if (interp_1 == 2 && interp_2 == 3) {
                        pred = (3*(*((T1*)s_data + s_id_1[1]))+6*(*((T1*)s_data + s_id_1[2])) - (*((T1*)s_data + s_id_1[3]))) / 8;
                        pred += (-(*((T1*)s_data + s_id_2[0]))+6*(*((T1*)s_data + s_id_2[1])) + 3*(*((T1*)s_data + s_id_2[2]))) / 8;
                        pred /= 2;
                    } else if (interp_1 == 2 && interp_2 == 2) {
                        pred = (3*(*((T1*)s_data + s_id_1[1]))+6*(*((T1*)s_data + s_id_1[2])) - (*((T1*)s_data + s_id_1[3]))) / 8;
                        pred += (3*(*((T1*)s_data + s_id_2[1]))+6*(*((T1*)s_data + s_id_2[2])) - (*((T1*)s_data + s_id_2[3]))) / 8;
                        pred /= 2;
                    } else if (interp_1 == 2 && interp_2 < 2) {
                        pred = (3*(*((T1*)s_data + s_id_1[1]))+6*(*((T1*)s_data + s_id_1[2])) - (*((T1*)s_data + s_id_1[3]))) / 8;
                    } else if (interp_1 <= 1 && interp_2 == 3) {
                        pred = (-(*((T1*)s_data + s_id_2[0]))+6*(*((T1*)s_data + s_id_2[1])) + 3*(*((T1*)s_data + s_id_2[2]))) / 8;
                    } else if (interp_1 <= 1 && interp_2 == 2) {
                        pred = (3*(*((T1*)s_data + s_id_2[1]))+6*(*((T1*)s_data + s_id_2[2])) - (*((T1*)s_data + s_id_2[3]))) / 8;
                    } else if (interp_1 == 1 && interp_2 == 1) {
                        pred = ((*((T1*)s_data + s_id_2[1]))+(*((T1*)s_data + s_id_2[2]))) / 2;
                        pred += ((*((T1*)s_data + s_id_1[1]))+(*((T1*)s_data + s_id_1[2]))) / 2;
                        pred /= 2;
                    } else if (interp_1 == 1 && interp_2 < 1) {
                        
                        pred = ((*((T1*)s_data + s_id_1[1]))+(*((T1*)s_data + s_id_1[2]))) / 2;
                    } else if (interp_1 == 0 && interp_2 == 1) {
                        pred = ((*((T1*)s_data + s_id_2[1]))+(*((T1*)s_data + s_id_2[2]))) / 2;
                    }
                    else{
                        pred = (*((T1*)s_data + s_id_1[1])) + (*((T1*)s_data + s_id_2[1])) - pred;
                    }
                    
            }

            if CONSTEXPR (CUBE) {  //
                auto interp_z = get_interp_order(z,BIZ,GDZ,global_z,data_size.z);
                auto interp_y = get_interp_order(y,BIY,GDY,global_y,data_size.y);
                auto interp_x = get_interp_order(x,BIX,GDX,global_x,data_size.x);
                
                #pragma unroll
                for(int id_itr = 0; id_itr < 4; ++id_itr){
                 tmp_x[id_itr] = s_data[z][y][id_x[id_itr]]; 
                }
                if(interp_z == 4){
                    #pragma unroll
                    for(int id_itr = 0; id_itr < 4; ++id_itr){
                        tmp_z[id_itr] = s_data[id_z[id_itr]][y][x];
                       }
                }
                if(interp_y == 4){
                    #pragma unroll
                    for(int id_itr = 0; id_itr < 4; ++id_itr){
                     tmp_y[id_itr] = s_data[z][id_y[id_itr]][x]; 
                    }
                }


                T1 pred_z[5], pred_y[5], pred_x[5];
                pred_x[0] = tmp_x[1];
                pred_x[1] = cubic_interpolator(tmp_x[0],tmp_x[1],tmp_x[2],tmp_x[3]);
                pred_x[2] = (-tmp_x[0]+6*tmp_x[1] + 3*tmp_x[2]) / 8;
                pred_x[3] = (3*tmp_x[1] + 6*tmp_x[2]-tmp_x[3]) / 8;
                pred_x[4] = (tmp_x[1] + tmp_x[2]) / 2;
                
                pred_y[1] = cubic_interpolator(tmp_y[0],tmp_y[1],tmp_y[2],tmp_y[3]);

                
                pred_z[1] = cubic_interpolator(tmp_z[0],tmp_z[1],tmp_z[2],tmp_z[3]);
                
                pred = pred_x[0];
                pred = (interp_z == 4 && interp_y == 4 && interp_x == 4) ? (pred_x[1] +  pred_y[1] + pred_z[1]) / 3 : pred;
                
                pred = (interp_z == 4 && interp_y == 4 && interp_x != 4) ? (pred_z[1] + pred_y[1]) / 2 : pred;
                pred = (interp_z == 4 && interp_y != 4 && interp_x == 4) ? (pred_z[1] + pred_x[1]) / 2 : pred;
                pred = (interp_z != 4 && interp_y == 4 && interp_x == 4) ? (pred_y[1] + pred_x[1]) / 2 : pred;
                
                pred = (interp_z == 4 && interp_y != 4 && interp_x != 4) ? pred_z[1]: pred;
                pred = (interp_z != 4 && interp_y == 4 && interp_x != 4) ? pred_y[1]: pred;
                pred = (interp_z != 4 && interp_y != 4 && interp_x == 4) ? pred_x[1]: pred;


                pred = (interp_z != 4 && interp_y != 4 && interp_x == 3) ? pred_x[2]: pred;
                pred = (interp_z != 4 && interp_y != 4 && interp_x == 2) ? pred_x[3]: pred;
                pred = (interp_z != 4 && interp_y != 4 && interp_x == 1) ? pred_x[4]: pred;
                // pred = (interp_z != 4 && interp_y != 4 && interp_x == 0) ? pred_x[0]: pred;
            }

            if CONSTEXPR (WORKFLOW == SPLINE3_COMPR) {
                
                auto          err = s_data[z][y][x] - pred;
                decltype(err) code;
                // TODO unsafe, did not deal with the out-of-cap case
                {
                    code = fabs(err) * eb_r + 1;
                    code = err < 0 ? -code : code;
                    code = int(code / 2) + radius;
                }
                s_ectrl[z][y][x] = code;  // TODO double check if unsigned type works
              
                s_data[z][y][x]  = pred + (code - radius) * ebx2;
                

            }
            else {  // TODO == DECOMPRESSS and static_assert

                
                auto code       = s_ectrl[z][y][x];
                s_data[z][y][x] = pred + (code - radius) * ebx2;
            }
        }
    };
    // -------------------------------------------------------------------------------- //

    if CONSTEXPR (COARSEN) {
        auto TOTAL = NUM_ELE;
        for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
            auto [x,y,z]    = xyzmap(_tix, unit);
            run(x, y, z);
        }
        
    }
    else {
        if(TIX<NUM_ELE){
            auto [x,y,z]    = xyzmap(TIX, unit);
            run(x, y, z);
        }
    }
    __syncthreads();
}