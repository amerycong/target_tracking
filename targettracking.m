%  Exploiting the Circulant Structure of Tracking-by-detection with Kernels
%
%  Main script for tracking, with a gaussian kernel.
%
%  João F. Henriques, 2012
%  http://www.isr.uc.pt/~henriques/


%choose the path to the videos (you'll be able to choose one with the GUI)
base_path = 'D:\Documents\MATLAB\target_tracking\';


%parameters according to the paper
padding = 1;					%extra area surrounding the target
output_sigma_factor = 1/16;		%spatial bandwidth (proportional to target)
sigma = 0.2;					%gaussian kernel bandwidth
lambda = 1e-2;					%regularization
interp_factor = 0.075;			%linear interpolation factor for adaptation



%notation: variables ending with f are in the frequency domain.

%ask the user for the video
video_path = choose_video(base_path);
if isempty(video_path), return, end  %user cancelled
[img_files, pos, target_sz, resize_image, ground_truth, video_path] = ...
    load_video_info(video_path);


%window size, taking padding into account
sz = floor(target_sz * (1 + padding));

%desired output (gaussian shaped), bandwidth proportional to target size
output_sigma = sqrt(prod(target_sz)) * output_sigma_factor;
[rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
y = exp(-0.5 / output_sigma^2 * (rs.^2 + cs.^2));
yf = fft2(y);

%store pre-computed cosine window
cos_window = hann(sz(1)) * hann(sz(2))';


time = 0;  %to calculate FPS
positions = zeros(numel(img_files), 2);  %to calculate precision

PSR_values = [];
pos_values = [];
occlusion_frames = [];


for frame = 1:numel(img_files),
    
    %for screenshot purposes
    occlude_flag = false;
    
    %load image
    im = imread([video_path img_files{frame}]);
    if size(im,3) > 1,
        im = rgb2gray(im);
    end
    if resize_image,
        im = imresize(im, 0.5);
    end
    
    tic()
    
    %extract and pre-process subwindow
    x = get_subwindow(im, pos, sz, cos_window);
    %default box color
    rect_color = 'g';
    
    if frame > 1,
        %calculate response of the classifier at all locations
        k = dense_gauss_kernel(sigma, x, z);
        response = real(ifft2(alphaf .* fft2(k)));   %(Eq. 9)
        
        %target location is at the maximum response
        [row, col] = find(response == max(response(:)), 1);
        pos = pos - floor(sz/2) + [row, col];
        pos_values(frame,:) = pos;
        %calculate PSR
        gmax = response(row,col);
        
        window_size = 11;
        sq_radius = (window_size-1)/2;
        %sidelobe area is target area - peak window area
        sidelobe = zeros(1,numel(response)-window_size^2);
        idx = 1;
        for r = 1:size(response,1)
            for c = 1:size(response,2)
                %sidelobe consists of values outside target window around max response
                if (r < row - sq_radius || r > row + sq_radius || c < col - sq_radius || c > col + sq_radius)
                    sidelobe(idx) = response(r,c);
                    idx = idx+1;
                end
            end
        end
        sidelobe_mean = mean(sidelobe);
        sidelobe_std = std(sidelobe);
        PSR = (response(row,col) - sidelobe_mean)/sidelobe_std;
        PSR_values(frame) = PSR;
        
        %prctile(PSR_values,2.5)
        PSR_threshold = 4.6;
        if PSR < PSR_threshold
            occlusion_frames = [occlusion_frames frame];
            occlude_flag = true;
            rect_color = 'r';
            %pause(0.5)
            fprintf('Occlusion detected in frame %d, PSR = %g\n',frame,PSR)
            %return

            %kalman filter prediction
            Nsamples = 20;
            dt = 1; %1 frame
            t=0:dt:dt*Nsamples;
            Vtrue_x = (pos_values(end-1,2)-pos_values(end-Nsamples,2))/(Nsamples-1);
            Vtrue_y = (pos_values(end-1,1)-pos_values(end-Nsamples,1))/(Nsamples-1);
            Xinitial = pos_values(end-Nsamples,2);
            Yinitial = pos_values(end-Nsamples,1);
            Xtrue = Xinitial + Vtrue_x * t;
            Ytrue = Yinitial + Vtrue_y * t;
            Xk_prev = [0;.5*Vtrue_x];
            Yk_prev = [0;.5*Vtrue_y];
            Xk = [];
            Yk = [];
            Phi = [1 dt;0  1];
            sigma_model = 1;
            P = [sigma_model^2 0;0 sigma_model^2];
            Q = [0 0; 0 0];
            M = [1 0];
            sigma_meas = 1; % 1 m/sec
            R = sigma_meas^2;
            
            Xk_buffer = zeros(2,Nsamples+1);
            Yk_buffer = zeros(2,Nsamples+1);
            Xk_buffer(:,1) = Xk_prev;
            Yk_buffer(:,1) = Yk_prev;
            Z_buffer_x = zeros(1,Nsamples+1);
            Z_buffer_y = zeros(1,Nsamples+1);
            for h=1:Nsamples
                
                % Z is the measurement vector. In our
                % case, Z = TrueData + RandomGaussianNoise
                Z_x = Xtrue(h+1)+sigma_meas*randn;
                Z_y = Ytrue(h+1)+sigma_meas*randn;
                Z_buffer_x(h+1) = Z_x;
                Z_buffer_y(h+1) = Z_y;
                
                % Kalman iteration
                P1 = Phi*P*Phi' + Q;
                S = M*P1*M' + R;
                
                % K is Kalman gain. If K is large, more weight goes to the measurement.
                % If K is low, more weight goes to the model prediction.
                K = P1*M'*inv(S);
                P = P1 - K*M*P1;
                
                Xk = Phi*Xk_prev + K*(Z_x-M*Phi*Xk_prev);
                Yk = Phi*Yk_prev + K*(Z_y-M*Phi*Yk_prev);
                Xk_buffer(:,h+1) = Xk;
                Yk_buffer(:,h+1) = Yk;
                
                % For the next iteration
                Xk_prev = Xk;
                Yk_prev = Yk;
            end
            pos = [Yk_buffer(1,end) Xk_buffer(1,end)];
            pos_values(frame,2) = pos(2);
            pos_values(frame,1) = pos(1);
        end
        
    end
    
    %get subwindow at current estimated target position, to train classifer
    x = get_subwindow(im, pos, sz, cos_window);
    
    %Kernel Regularized Least-Squares, calculate alphas (in Fourier domain)
    k = dense_gauss_kernel(sigma, x);
    new_alphaf = yf ./ (fft2(k) + lambda);   %(Eq. 7)
    new_z = x;
    
    if frame == 1,  %first frame, train with a single image
        alphaf = new_alphaf;
        z = x;
    else
        %subsequent frames, interpolate model
        alphaf = (1 - interp_factor) * alphaf + interp_factor * new_alphaf;
        z = (1 - interp_factor) * z + interp_factor * new_z;
    end
    
    %save position and calculate FPS
    positions(frame,:) = pos;
    time = time + toc();
    
    %visualization
    rect_position = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
    if frame == 1,  %first frame, create GUI
        figure('IntegerHandle','off', 'Name',['Tracker - ' video_path])
        im = rgb2gray(insertText(im,[0 0],frame));
        im_handle = imshow(im, 'Border','tight', 'InitialMag',200);
        rect_handle = rectangle('Position',rect_position,'EdgeColor',rect_color);
    else
        try  %subsequent frames, update GUI
            im = rgb2gray(insertText(im,[0 0],frame));
            set(im_handle, 'CData', im)
            set(rect_handle, 'Position', rect_position, 'EdgeColor',rect_color)
        catch  %#ok, user has closed the window
            return
        end
    end
    
    drawnow
    %pause(0.05)  %uncomment to run slower
end

if resize_image, positions = positions * 2; end

disp(['Frames-per-second: ' num2str(numel(img_files) / time)])

%show the precisions plot
show_precision(positions, ground_truth, video_path)

